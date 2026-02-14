import React, { useState } from 'react';
import { systemApi } from '../lib/api';

export const EmergencyStop = () => {
    const [triggered, setTriggered] = useState(false);

    const handleStop = async () => {
        // Optimistic UI update
        setTriggered(true);

        try {
            // Spam the request to ensure it gets through
            // We want it out FAST
            const req = systemApi.emergencyStop();

            // Visual feedback
            setTimeout(() => setTriggered(false), 2000);

            await req;
            console.log("EMERGENCY STOP SENT");
        } catch (e) {
            console.error("EMERGENCY STOP FAILED", e);
            alert("EMERGENCY STOP FAILED TO SEND! HIT PHYSICAL KILL SWITCH!");
        }
    };

    return (
        <button
            onClick={handleStop}
            className={`
                group z-[60]
                flex items-center justify-center
                px-4 py-1.5 rounded-full
                border border-red-500/30 dark:border-red-500/40
                shadow-sm backdrop-blur-md
                transition-all duration-200 active:scale-95
                ${triggered
                    ? 'bg-red-600 text-white shadow-red-500/50 scale-95'
                    : 'bg-red-500/10 dark:bg-red-500/20 text-red-600 dark:text-red-400 hover:bg-red-500 hover:text-white hover:shadow-red-500/40'
                }
            `}
            title="EMERGENCY STOP - DISABLE ALL TORQUE"
        >
            <div className="flex items-center gap-2">
                <div className={`
                    w-1.5 h-1.5 rounded-full bg-current
                    ${triggered ? 'animate-ping' : ''}
                `} />
                <span className="font-bold tracking-widest text-[10px]">STOP</span>
            </div>
        </button>
    );
};
