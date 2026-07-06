"""
Krea2ReferenceLock (By Fedor) - v2 "latent mold" reference steering.

Sculptor's-mold guidance: the reference image is VAE-encoded into a latent
"cast". At every sampling step (within a scheduled window) the model's
predicted-clean latent (x0 / denoised) is compared against the cast inside the
target bounding box and nudged toward it:

    denoised[box] += strength * mask * (mold - denoised[box])

Krea2 has NO native reference-latent pathway (its DiT sequence is strictly
[text | image] and extra_conds discards reference_latents), so this operates
one layer up - at the sampler's post-CFG hook - which is model-agnostic and
never touches model weights (fp8-safe). Same intervention family as latent-
anchor nodes for LTX2, adapted to Krea2 stills + explicit bboxes.

Tier-1 characteristics:
  * Anchors composition AND identity inside the box toward the reference.
  * Couples pose/framing to the reference (the box tends to inherit the
    reference's pose). Use start/end_percent to limit this: guiding only the
    early-mid steps sets structure, then releases the model to integrate.
  * Stacks cleanly with Krea2RegionalMultiLoRA (different intervention
    points: LoRA delta injection is inside the forward; this is post-CFG).

Chainable: each node instance guides one box; wire model through several
Krea2ReferenceLock nodes for multiple references.
"""

import logging

import torch
import torch.nn.functional as F

from .krea2_regional_multilora import (
    _coerce_bbox_norm,
    _normalize_bboxes,
    _rect_token_mask,
)


def _latent_rect_mask(rows, cols, box, feather, device, dtype):
    """Feathered 2D mask (rows x cols) for a normalised (x0,y0,x1,y1) box."""
    x0, y0, x1, y1 = box
    m = _rect_token_mask(rows, cols, x0, y0, x1, y1, feather)
    return m.reshape(1, 1, rows, cols).to(device=device, dtype=dtype)


class Krea2ReferenceLock:
    """Steers the in-progress denoised latent toward a VAE-encoded reference
    image inside a bounding box (post-CFG guidance; no weight changes)."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "vae": ("VAE",),
                "reference_image": ("IMAGE", {
                    "tooltip": "The 'mold': what this box should converge toward.",
                }),
                "strength": ("FLOAT", {
                    "default": 0.30, "min": 0.0, "max": 1.0, "step": 0.01,
                    "tooltip": (
                        "Per-step pull toward the reference inside the box. "
                        "0.2-0.4 = strong anchor that still integrates with the scene; "
                        "0.7+ approaches a paste."
                    ),
                }),
                "start_percent": ("FLOAT", {
                    "default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01,
                    "tooltip": "Start of the guidance window (fraction of sampling).",
                }),
                "end_percent": ("FLOAT", {
                    "default": 0.60, "min": 0.0, "max": 1.0, "step": 0.01,
                    "tooltip": (
                        "End of the guidance window. Ending around 0.5-0.7 locks "
                        "structure/identity early, then lets the model blend seams "
                        "and lighting naturally in the remaining steps."
                    ),
                }),
                "feather": ("FLOAT", {
                    "default": 0.06, "min": 0.0, "max": 0.5, "step": 0.01,
                    "tooltip": "Soft edge of the guidance mask (fraction of latent grid).",
                }),
                "canvas_width": ("INT", {
                    "default": 1024, "min": 64, "max": 16384, "step": 16,
                    "tooltip": "Canvas width used to interpret pixel-space bboxes.",
                }),
                "canvas_height": ("INT", {
                    "default": 1024, "min": 64, "max": 16384, "step": 16,
                }),
            },
            "optional": {
                "bboxes": ("BOUNDING_BOX", {
                    "tooltip": "Boxes from the same builder feeding the MultiLoRA node.",
                }),
                "box_index": ("INT", {
                    "default": 0, "min": 0, "max": 63,
                    "tooltip": "Which wired box this reference locks onto (0-based).",
                }),
                "box_x0": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01,
                                     "tooltip": "Manual box (normalised) when no bboxes are wired."}),
                "box_y0": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "box_x1": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "box_y1": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
            },
        }

    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "apply"
    CATEGORY = "Krea2/By Fedor"

    DESCRIPTION = (
        "Krea2 Reference Lock (By Fedor, v2 beta). Latent-mold guidance: encodes a "
        "reference image and nudges the in-progress denoised latent toward it inside "
        "a bounding box, each step, over a scheduled window. Anchors identity/"
        "composition without any model-weight changes. Chain one node per box."
    )

    def apply(self, model, vae, reference_image, strength, start_percent,
              end_percent, feather, canvas_width, canvas_height,
              bboxes=None, box_index=0, box_x0=0.0, box_y0=0.0,
              box_x1=1.0, box_y1=1.0):
        if strength <= 0.0 or end_percent <= start_percent:
            logging.info("[Krea2ReferenceLock] disabled (strength/window); passthrough.")
            return (model,)

        # Resolve the target box (normalised 0..1).
        box = None
        frame = _normalize_bboxes(bboxes)
        if frame:
            if box_index < len(frame):
                box = _coerce_bbox_norm(frame[box_index], int(canvas_width), int(canvas_height))
            else:
                logging.warning("[Krea2ReferenceLock] box_index %d out of range (%d boxes); "
                                "using manual box.", box_index, len(frame))
        if box is None:
            box = (min(box_x0, box_x1), min(box_y0, box_y1),
                   max(box_x0, box_x1), max(box_y0, box_y1))
        if box[2] - box[0] < 1e-3 or box[3] - box[1] < 1e-3:
            logging.warning("[Krea2ReferenceLock] degenerate box %s; passthrough.", box)
            return (model,)

        # Encode the mold. VAE latent is in storage space; sampling runs in the
        # model's processed space, so convert with process_latent_in.
        pixels = reference_image[:, :, :, :3]
        ref_storage = vae.encode(pixels)
        m = model.clone()
        ref_model_space = m.model.process_latent_in(ref_storage)

        # Guidance window in sigma terms (sigma decreases during sampling).
        ms = m.get_model_object("model_sampling")
        sigma_start = ms.percent_to_sigma(float(start_percent))
        sigma_end = ms.percent_to_sigma(float(end_percent))

        state = {"mold": None, "mask": None, "shape": None, "logged": False}
        fth = float(feather)
        w = float(strength)

        def post_cfg(args):
            denoised = args["denoised"]
            sigma = args["sigma"]
            sv = float(sigma.max().item()) if torch.is_tensor(sigma) else float(sigma)
            # Inside the window? (sigma runs high -> low.)
            if sv > sigma_start + 1e-9 or sv < sigma_end - 1e-9:
                return denoised
            if denoised.dim() != 4:
                return denoised

            B, C, H, Wd = denoised.shape
            if state["shape"] != (C, H, Wd):
                # Build the mold canvas + mask once per resolution.
                x0, y0, x1, y1 = box
                bx0, bx1 = int(round(x0 * Wd)), int(round(x1 * Wd))
                by0, by1 = int(round(y0 * H)), int(round(y1 * H))
                bx1, by1 = max(bx1, bx0 + 1), max(by1, by0 + 1)
                bx1, by1 = min(bx1, Wd), min(by1, H)
                ref = ref_model_space.to(device=denoised.device, dtype=torch.float32)
                if ref.shape[1] != C:
                    logging.warning("[Krea2ReferenceLock] ref latent has %d channels, "
                                    "latent has %d; passthrough.", ref.shape[1], C)
                    state["shape"] = (C, H, Wd)
                    state["mold"] = None
                    return denoised
                fitted = F.interpolate(ref[:1], size=(by1 - by0, bx1 - bx0),
                                       mode="bilinear", align_corners=False)
                mold = torch.zeros(1, C, H, Wd, device=denoised.device, dtype=torch.float32)
                mold[:, :, by0:by1, bx0:bx1] = fitted
                state["mold"] = mold
                state["mask"] = _latent_rect_mask(H, Wd, box, fth,
                                                  denoised.device, torch.float32)
                state["shape"] = (C, H, Wd)
                if not state["logged"]:
                    logging.info("[Krea2ReferenceLock] mold armed: latent %dx%d, box px "
                                 "(%d,%d)-(%d,%d), window sigma %.4f->%.4f, strength %.2f",
                                 Wd, H, bx0, by0, bx1, by1, sigma_start, sigma_end, w)
                    state["logged"] = True

            if state["mold"] is None:
                return denoised
            d32 = denoised.float()
            steered = d32 + (w * state["mask"]) * (state["mold"] - d32)
            return steered.to(denoised.dtype)

        m.set_model_sampler_post_cfg_function(post_cfg)
        return (m,)


NODE_CLASS_MAPPINGS = {
    "Krea2ReferenceLock": Krea2ReferenceLock,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Krea2ReferenceLock": "Krea2 Reference Lock — Latent Mold (By Fedor)",
}
