import type { ResizableModalProps } from "../../../lib/api/types";

export interface DeployModalProps extends ResizableModalProps {}

export type DeployPhase = "setup" | "running";

export type DeployModeLabel = "Inference" | "HIL" | "HIL-SERL";
