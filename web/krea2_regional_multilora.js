// Krea2 Regional Multi-LoRA - dynamic region rows.
//
// Adds an "+ Add Region" button and, per region, a LoRA combo + strength +
// enable toggle + remove button. All rows are serialized into the node's
// `regions_json` widget (the Python side's source of truth), so the graph
// still round-trips through save/load and the API.
//
// Auto-sync: when a Prompt Builder node is wired into the bboxes input, the
// region row count automatically follows the number of boxes drawn there.
// Existing LoRA assignments are preserved; only the count changes.

import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

const NODE_TYPE = "Krea2RegionalMultiLoRA";
const JSON_WIDGET = "regions_json";

let LORA_LIST = ["None"];

async function ensureLoraList() {
  if (LORA_LIST.length > 1) return LORA_LIST;
  try {
    const resp = await api.fetchApi("/object_info/LoraLoader");
    const info = await resp.json();
    const names = info?.LoraLoader?.input?.required?.lora_name?.[0];
    if (Array.isArray(names) && names.length) {
      LORA_LIST = ["None", ...names.filter((n) => n !== "None")];
    }
  } catch (e) {
    console.warn("[Krea2RegionalMultiLoRA] could not fetch lora list:", e);
  }
  return LORA_LIST;
}

function readRegions(node) {
  const w = node.widgets?.find((x) => x.name === JSON_WIDGET);
  if (!w) return [];
  try {
    const parsed = JSON.parse(w.value || "[]");
    return Array.isArray(parsed) ? parsed : [];
  } catch (e) {
    return [];
  }
}

function writeRegions(node, regions) {
  const w = node.widgets?.find((x) => x.name === JSON_WIDGET);
  if (!w) return;
  w.value = JSON.stringify(regions, null, 2);
  if (w.inputEl) w.inputEl.value = w.value;
}

function markTransient(w) {
  w.__k2region = true;
  w.serialize = false;
  if (!w.options) w.options = {};
  w.options.serialize = false;
  return w;
}

// ---------------------------------------------------------------------------
// Bbox auto-sync
// ---------------------------------------------------------------------------

// Return the number of boxes in the node wired to our bboxes input, or null
// if nothing is wired / no box source can be found.
//
// The Ideogram4PromptBuilderKJ keeps a LIVE array on the node (`_boxes`) that
// updates the instant a box is created or deleted - and is length 0 when the
// last box is removed. Reading it directly is what makes the sync immediate and
// lets us detect the drop to zero (the serialized STRING widget becomes "" at
// zero boxes, which is why the old string-parsing path could never sync down).
function getBboxCount(node) {
  const bboxInput = node.inputs?.find((i) => i.name === "bboxes");
  if (!bboxInput || bboxInput.link == null) return null;

  const linkInfo = node.graph?.links?.[bboxInput.link];
  if (!linkInfo) return null;

  const srcNode = node.graph?.getNodeById(linkInfo.origin_id);
  if (!srcNode) return null;

  // Preferred: the builder's live box array.
  if (Array.isArray(srcNode._boxes)) {
    return srcNode._boxes.length;
  }

  // Fallback: a STRING widget holding an array of box objects (other builders).
  for (const w of srcNode.widgets || []) {
    if (typeof w.value !== "string") continue;
    const s = w.value.trim();
    if (s === "") continue;
    try {
      const parsed = JSON.parse(s);
      if (
        Array.isArray(parsed) &&
        parsed.length > 0 &&
        typeof parsed[0] === "object" &&
        ("x" in parsed[0] || "x0" in parsed[0] || "width" in parsed[0] || "w" in parsed[0])
      ) {
        return parsed.length;
      }
    } catch (_) {}
  }
  return null;
}

// Adjust the regions array so its length matches targetCount, preserving all
// existing LoRA assignments. New rows get sensible defaults.
function syncRegionCount(node, targetCount) {
  const regions = readRegions(node);
  if (regions.length === targetCount) return; // nothing to do

  if (regions.length < targetCount) {
    while (regions.length < targetCount) {
      regions.push({ lora: "None", strength: 1.1, enable: true });
    }
  } else {
    regions.splice(targetCount);
  }

  writeRegions(node, regions);
  rebuildRows(node);
}

// Single entry point used by every trigger (draw / connect / global mouse+key).
// Only acts when the box count actually changed, and refuses to wipe existing
// rows to zero during the brief post-load settling window (when a connected
// builder may not have restored its boxes yet).
function checkAndSync(node) {
  const count = getBboxCount(node);
  if (count === null) return;
  if (count === node.__k2lastBboxCount) return;

  if (
    count === 0 &&
    readRegions(node).length > 0 &&
    node.__k2loadGuardUntil &&
    Date.now() < node.__k2loadGuardUntil
  ) {
    return; // load-race guard: don't clear rows before the builder restores
  }

  node.__k2lastBboxCount = count;
  syncRegionCount(node, count);
}

// Global backstop: box edits happen inside the builder's own canvas/dock, which
// doesn't reliably repaint OUR node. Every builder change ends on a mouseup
// (it dispatches one on commit), and deletions come via Del/Backspace, so we
// re-check all our nodes on those events. Registered once per page.
function installGlobalResync(app) {
  if (window.__k2RegionSyncHooked) return;
  window.__k2RegionSyncHooked = true;
  const resyncAll = () => {
    const nodes = app.graph?._nodes || [];
    for (const n of nodes) {
      if (n.type === NODE_TYPE) checkAndSync(n);
    }
  };
  window.addEventListener("mouseup", () => setTimeout(resyncAll, 0), true);
  window.addEventListener(
    "keyup",
    (e) => {
      if (e.key === "Delete" || e.key === "Backspace") setTimeout(resyncAll, 0);
    },
    true
  );
}

// ---------------------------------------------------------------------------
// Row rendering
// ---------------------------------------------------------------------------

function rebuildRows(node) {
  if (node.widgets) {
    node.widgets = node.widgets.filter((w) => !w.__k2region);
  }

  const regions = readRegions(node);

  regions.forEach((region, idx) => {
    const enableW = node.addWidget(
      "toggle",
      `region ${idx + 1} enabled`,
      region.enable !== false,
      (v) => {
        const r = readRegions(node);
        if (r[idx]) { r[idx].enable = v; writeRegions(node, r); }
      },
      { on: "on", off: "off" }
    );
    markTransient(enableW);

    const loraW = node.addWidget(
      "combo",
      `region ${idx + 1} lora`,
      region.lora || "None",
      (v) => {
        const r = readRegions(node);
        if (r[idx]) { r[idx].lora = v; writeRegions(node, r); }
      },
      { values: LORA_LIST }
    );
    markTransient(loraW);

    const strW = node.addWidget(
      "number",
      `region ${idx + 1} strength`,
      typeof region.strength === "number" ? region.strength : 1.1,
      (v) => {
        const r = readRegions(node);
        if (r[idx]) { r[idx].strength = v; writeRegions(node, r); }
      },
      { min: -10.0, max: 10.0, step: 0.1, precision: 2 }
    );
    markTransient(strW);

    const rmW = node.addWidget("button", `  remove region ${idx + 1}`, null, () => {
      const r = readRegions(node);
      r.splice(idx, 1);
      writeRegions(node, r);
      rebuildRows(node);
      node.setDirtyCanvas(true, true);
    });
    markTransient(rmW);
  });

  node.setDirtyCanvas(true, true);
}

// ---------------------------------------------------------------------------
// Extension registration
// ---------------------------------------------------------------------------

app.registerExtension({
  name: "fedor.Krea2RegionalMultiLoRA",
  async beforeRegisterNodeDef(nodeType, nodeData) {
    if (nodeData.name !== NODE_TYPE) return;
    await ensureLoraList();
    installGlobalResync(app);

    const onNodeCreated = nodeType.prototype.onNodeCreated;
    nodeType.prototype.onNodeCreated = function () {
      const r = onNodeCreated ? onNodeCreated.apply(this, arguments) : undefined;
      this.__k2lastBboxCount = null;

      const addBtn = this.addWidget("button", "+ Add Region", null, () => {
        const regions = readRegions(this);
        regions.push({ lora: "None", strength: 1.1, enable: true });
        writeRegions(this, regions);
        rebuildRows(this);
      });
      addBtn.__k2add = true;

      rebuildRows(this);
      return r;
    };

    const onConfigure = nodeType.prototype.onConfigure;
    nodeType.prototype.onConfigure = function (o) {
      const r = onConfigure ? onConfigure.apply(this, arguments) : undefined;
      this.__k2lastBboxCount = null;
      // Protect the restored rows from being cleared to zero while a connected
      // builder is still restoring its own boxes during graph load.
      this.__k2loadGuardUntil = Date.now() + 2500;
      setTimeout(() => rebuildRows(this), 0);
      return r;
    };

    // onConnectionsChange fires when the bboxes wire is connected or removed.
    // Do an immediate sync so the row count adjusts without waiting for a draw.
    const onConnectionsChange = nodeType.prototype.onConnectionsChange;
    nodeType.prototype.onConnectionsChange = function (type, index, connected, link_info) {
      const r = onConnectionsChange
        ? onConnectionsChange.apply(this, arguments)
        : undefined;

      const bboxIdx = this.inputs?.findIndex((i) => i.name === "bboxes");
      if (index === bboxIdx) {
        this.__k2lastBboxCount = null; // force re-check
        // A fresh manual wire is a deliberate user action, not load - drop the
        // guard so connecting an empty builder can still show zero rows.
        if (connected) this.__k2loadGuardUntil = 0;
        setTimeout(() => checkAndSync(this), 50);
      }
      return r;
    };

    // onDrawForeground is one of several triggers (see installGlobalResync).
    // The __k2lastBboxCount cache means we only act when the count actually
    // changes, so the per-frame cost is just a couple of property lookups.
    const onDrawForeground = nodeType.prototype.onDrawForeground;
    nodeType.prototype.onDrawForeground = function (ctx) {
      if (onDrawForeground) onDrawForeground.apply(this, arguments);
      checkAndSync(this);
    };
  },
});
