export interface HILModalProps {
    isOpen: boolean;
    onClose: () => void;
    maximizedWindow: string | null;
    setMaximizedWindow: (window: string | null) => void;
}

export interface Policy {
    id: string;
    name: string;
    policy_type: string;
    status: string;
    checkpoint_path: string;
}

export interface HILStatus {
    active: boolean;
    mode: 'idle' | 'autonomous' | 'human' | 'paused';
    policy_id: string;
    intervention_dataset: string;
    task: string;
    episode_active: boolean;
    episode_count: number;
    intervention_count: number;
    current_episode_interventions: number;
    autonomous_frames: number;
    human_frames: number;
    // Policy configuration - which cameras/arms the policy was trained on
    policy_config?: {
        cameras: string[];
        arms: string[];
        type: string;
    };
    // Safety settings
    movement_scale?: number;
}
