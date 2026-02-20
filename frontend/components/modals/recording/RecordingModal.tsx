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
            // Map pairing names → follower IDs
            const { pairings } = await armsApi.listPairings();
            const followerIds = pairings
                .filter((p) => selectedPairings.includes(p.name))
                .map((p) => p.follower_id);

            const data = await recordingApi.startSession({
                repo_id: datasetConfig.repo_id,
                task: datasetConfig.task,
                selected_cameras: selectedCameras.length > 0 ? selectedCameras : null,
                selected_arms: followerIds.length > 0 ? followerIds : null,
            });

            if (data.status === 'success') {
                // Start teleop alongside recording
                await teleopApi.start({ force: false });
                // Transition to active recording view
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
