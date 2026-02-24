import { useState, useEffect } from 'react';
import RecordingSetupPanel from './RecordingSetupPanel';
import { recordingApi, teleopApi, armsApi } from '../../../lib/api';

interface RecordingModalProps {
    isOpen: boolean;
    onClose: () => void;
    maximizedWindow: string | null;
    setMaximizedWindow: (window: string | null) => void;
    onSessionStarted?: (datasetName: string) => void;
}

export default function RecordingModal({
    isOpen,
    onClose,
    onSessionStarted,
}: RecordingModalProps) {
    // Setup state
    const [datasetConfig, setDatasetConfig] = useState({
        repo_id: 'lerobot/dataset',
        task: 'Pick up the cube',
    });
    const [selectedPairings, setSelectedPairings] = useState<string[]>([]);
    const [selectedCameras, setSelectedCameras] = useState<string[]>([]);
    const [error, setError] = useState('');
    const [isStarting, setIsStarting] = useState(false);

    // Check if a session is already active on open → jump to active view
    useEffect(() => {
        if (!isOpen) return;
        recordingApi.status().then((data) => {
            if (data.session_active) {
                onSessionStarted?.(datasetConfig.repo_id);
            }
        }).catch(console.error);
    }, [isOpen]); // eslint-disable-line react-hooks/exhaustive-deps

    const handleStartSession = async () => {
        setError('');
        setIsStarting(true);

        try {
            // Map pairing names → leader + follower IDs
            const { pairings } = await armsApi.listPairings();
            const matchedPairings = pairings.filter((p) => selectedPairings.includes(p.name));
            const followerIds = matchedPairings.map((p) => p.follower_id);
            // Teleop needs ALL arm IDs (both leaders and followers) to build pairing contexts
            const activeArms = matchedPairings.flatMap((p) => [p.leader_id, p.follower_id]);

            // 1. Ensure teleop is running (builds pairing contexts needed by recording)
            //    CRITICAL: Do NOT call start() if already running — a stop→restart
            //    cycle re-enables motors with full MIT gains against stale position
            //    error, causing dangerous torque spikes / vibration.
            const teleopStatus = await teleopApi.status();
            if (!teleopStatus.running) {
                const teleopResult = await teleopApi.start({ force: false, active_arms: activeArms });
                if ((teleopResult as any).status === 'error') {
                    setError((teleopResult as any).message || 'Failed to start teleop');
                    setIsStarting(false);
                    return;
                }
            }

            // 2. THEN start recording session (needs pairing contexts from teleop)
            const data = await recordingApi.startSession({
                repo_id: datasetConfig.repo_id,
                task: datasetConfig.task,
                selected_cameras: selectedCameras.length > 0 ? selectedCameras : null,
                selected_arms: followerIds.length > 0 ? followerIds : null,
            });

            if (data.status === 'success') {
                onSessionStarted?.(datasetConfig.repo_id);
            } else {
                setError(data.message);
            }
        } catch (e: unknown) {
            const msg = e instanceof Error ? e.message : 'Failed to start session';
            setError(msg);
        }

        setIsStarting(false);
    };

    if (!isOpen) return null;

    return (
        <RecordingSetupPanel
            isOpen={true}
            onClose={onClose}
            onStartSession={handleStartSession}
            datasetConfig={datasetConfig}
            setDatasetConfig={setDatasetConfig}
            selectedPairings={selectedPairings}
            setSelectedPairings={setSelectedPairings}
            selectedCameras={selectedCameras}
            setSelectedCameras={setSelectedCameras}
            error={error}
            isStarting={isStarting}
        />
    );
}
