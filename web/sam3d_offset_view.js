import { app } from "../../../scripts/app.js";

const EXTENSION_FOLDER = (() => {
    const url = import.meta.url;
    const match = url.match(/\/extensions\/([^/]+)\//);
    return match ? match[1] : "sam3d-body-comfyUI-camshottoolkit";
})();

const STATE_WIDGET = "interactive_state";

function getWidget(node, name) {
    return node.widgets?.find((widget) => widget.name === name) || null;
}

function getWidgetValue(node, name, fallback = null) {
    const widget = getWidget(node, name);
    return widget ? widget.value : fallback;
}

function setWidgetValue(node, name, value) {
    const widget = getWidget(node, name);
    if (!widget) {
        return;
    }
    widget.value = value;
    widget.callback?.(value);
}

function markNodeChanged(node) {
    node.setDirtyCanvas?.(true, true);
    node.graph?.afterChange?.();
    app.graph?.afterChange?.();
    app.graph?.setDirtyCanvas?.(true, true);
}

function extractSam3dUi(message) {
    const out = message?.output;
    if (out && typeof out === "object" && (out.preview_mesh || out.active_camera_state)) {
        return out;
    }
    if (out && typeof out === "object" && out.output && typeof out.output === "object" && (out.output.preview_mesh || out.output.active_camera_state)) {
        return out.output;
    }
    if (message && typeof message === "object" && (message.preview_mesh || message.active_camera_state)) {
        return message;
    }
    return null;
}

function ensureWidgetDefaults(node) {
    const boolDefaults = [
        ["enable_viewer", true],
        ["use_interactive_view", true],
        ["show_viewer_hud", true],
    ];
    for (const [name, fallback] of boolDefaults) {
        const value = getWidgetValue(node, name, fallback);
        if (typeof value !== "boolean") {
            setWidgetValue(node, name, fallback);
        }
    }
}

function buildViewerState(node) {
    return {
        nodeId: node.id,
        enable_viewer: !!getWidgetValue(node, "enable_viewer", true),
        use_interactive_view: !!getWidgetValue(node, "use_interactive_view", true),
        show_viewer_hud: !!getWidgetValue(node, "show_viewer_hud", true),
        interactive_state: String(getWidgetValue(node, STATE_WIDGET, "") || ""),
        mesh_r: Number(getWidgetValue(node, "mesh_r", 235) ?? 235),
        mesh_g: Number(getWidgetValue(node, "mesh_g", 235) ?? 235),
        mesh_b: Number(getWidgetValue(node, "mesh_b", 235) ?? 235),
        bg_preset: String(getWidgetValue(node, "bg_preset", "mid_gray") || "mid_gray"),
        bg_r: Number(getWidgetValue(node, "bg_r", 38) ?? 38),
        bg_g: Number(getWidgetValue(node, "bg_g", 38) ?? 38),
        bg_b: Number(getWidgetValue(node, "bg_b", 38) ?? 38),
        lighting_preset: String(getWidgetValue(node, "lighting_preset", "studio") || "studio"),
        ambient_intensity: Number(getWidgetValue(node, "ambient_intensity", 0.35) ?? 0.35),
        key_intensity: Number(getWidgetValue(node, "key_intensity", 14.0) ?? 14.0),
        key_yaw: Number(getWidgetValue(node, "key_yaw", 35.0) ?? 35.0),
        key_pitch: Number(getWidgetValue(node, "key_pitch", 35.0) ?? 35.0),
        fill_intensity: Number(getWidgetValue(node, "fill_intensity", 6.0) ?? 6.0),
        rim_intensity: Number(getWidgetValue(node, "rim_intensity", 8.0) ?? 8.0),
    };
}

app.registerExtension({
    name: "sam3d.offset.viewer",

    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name !== "CamShotToolkitRenderOffsetView") {
            return;
        }

        const onNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            const result = onNodeCreated ? onNodeCreated.apply(this, arguments) : undefined;

            const stateWidget = getWidget(this, STATE_WIDGET);
            if (stateWidget) {
                stateWidget.hidden = true;
            }
            ensureWidgetDefaults(this);

            const container = document.createElement("div");
            container.style.width = "100%";
            container.style.height = "100%";
            container.style.minHeight = "420px";
            container.style.display = "flex";
            container.style.flexDirection = "column";
            container.style.overflow = "hidden";
            container.style.background = "#101010";

            this.addDOMWidget("sam3d_offset_viewer", "SAM3D_OFFSET_VIEWER", container, {
                getValue() {
                    return "";
                },
                setValue() {},
            });

            const viewerWidget = this.widgets?.find((w) => w.name === "sam3d_offset_viewer");
            if (viewerWidget && this.widgets) {
                const i = this.widgets.indexOf(viewerWidget);
                if (i > 0) {
                    this.widgets.splice(i, 1);
                    this.widgets.unshift(viewerWidget);
                }
            }
            this.setDirtyCanvas?.(true, true);

            this.sam3dIframe = null;
            this.sam3dUi = null;
            this.sam3dLoaded = false;
            this.sam3dLastSignature = "";

            const setViewerMounted = (enabled) => {
                container.style.display = enabled ? "flex" : "none";
                container.style.minHeight = enabled ? "420px" : "0";
                container.style.height = enabled ? "100%" : "0";
                if (viewerWidget) {
                    viewerWidget.hidden = !enabled;
                }
                this.setDirtyCanvas?.(true, true);
            };

            const unloadViewer = () => {
                const existing = this.sam3dIframe;
                this.sam3dLoaded = false;
                this.sam3dLastSignature = "";
                if (!existing) {
                    setViewerMounted(false);
                    return;
                }
                try {
                    existing.src = "about:blank";
                } catch (_) {}
                existing.remove();
                this.sam3dIframe = null;
                setViewerMounted(false);
            };

            const mountViewer = () => {
                if (this.sam3dIframe) {
                    setViewerMounted(true);
                    return;
                }
                const iframe = document.createElement("iframe");
                iframe.style.width = "100%";
                iframe.style.height = "100%";
                iframe.style.border = "none";
                iframe.style.background = "#101010";
                iframe.src = `/extensions/${EXTENSION_FOLDER}/viewer_sam3d_offset.html?v=${Date.now()}`;
                iframe.addEventListener("load", () => {
                    if (this.sam3dIframe !== iframe) {
                        return;
                    }
                    this.sam3dLoaded = true;
                    this.sam3dSendState(true);
                });
                this.sam3dIframe = iframe;
                container.appendChild(iframe);
                setViewerMounted(true);
            };

            this.sam3dSyncViewerEnabled = () => {
                const enabled = !!getWidgetValue(this, "enable_viewer", true);
                if (enabled) {
                    mountViewer();
                } else {
                    unloadViewer();
                }
            };

            this.sam3dSendState = (force = false) => {
                const enabled = !!getWidgetValue(this, "enable_viewer", true);
                const iframe = this.sam3dIframe;
                if (!enabled || !this.sam3dLoaded || !iframe?.contentWindow) {
                    return;
                }
                const state = buildViewerState(this);
                const ui = this.sam3dUi || {};
                const signature = JSON.stringify({ ui, state });
                if (!force && signature === this.sam3dLastSignature) {
                    return;
                }
                this.sam3dLastSignature = signature;
                iframe.contentWindow.postMessage({ type: "SAM3D_OFFSET_STATE", ui, state }, "*");
            };

            this.sam3dSyncViewerEnabled();
            this.sam3dPoll = setInterval(() => {
                this.sam3dSyncViewerEnabled?.();
                this.sam3dSendState(false);
            }, 150);

            const onRemoved = this.onRemoved;
            this.onRemoved = function () {
                clearInterval(this.sam3dPoll);
                this.sam3dSyncViewerEnabled = null;
                this.sam3dLoaded = false;
                this.sam3dLastSignature = "";
                if (this.sam3dIframe) {
                    try {
                        this.sam3dIframe.src = "about:blank";
                    } catch (_) {}
                    this.sam3dIframe.remove();
                    this.sam3dIframe = null;
                }
                if (this.sam3dWindowMessageHandler) {
                    window.removeEventListener("message", this.sam3dWindowMessageHandler);
                    this.sam3dWindowMessageHandler = null;
                }
                return onRemoved ? onRemoved.apply(this, arguments) : undefined;
            };

            const onExecuted = this.onExecuted;
            this.onExecuted = function (message) {
                onExecuted?.apply(this, arguments);
                const ui = extractSam3dUi(message);
                if (!ui) {
                    return;
                }
                this.sam3dUi = ui;
                this.sam3dSendState(true);
            };

            this.sam3dWindowMessageHandler = (event) => {
                const data = event?.data;
                if (!data || data.type !== "SAM3D_OFFSET_VIEWER_STATE") {
                    return;
                }
                if (Number(data.nodeId) !== Number(this.id)) {
                    return;
                }
                if (typeof data.interactive_state !== "string") {
                    return;
                }
                setWidgetValue(this, STATE_WIDGET, data.interactive_state);
                setWidgetValue(this, "use_interactive_view", true);
                markNodeChanged(this);
            };
            window.addEventListener("message", this.sam3dWindowMessageHandler);

            return result;
        };
    },
});
