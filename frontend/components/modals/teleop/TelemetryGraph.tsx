import React from 'react';
import { LineChart, Line, XAxis, YAxis, Tooltip, ResponsiveContainer } from 'recharts';

// Use a fixed list of colors for lines
const colors = ["#8884d8", "#82ca9d", "#ffc658", "#ff7300", "#00C49F"];

interface TelemetryGraphProps {
    graphData: any[];
    torqueData: any;
    isRunning: boolean;
}

export default function TelemetryGraph({ graphData, torqueData, isRunning }: TelemetryGraphProps) {
    return (
        <>
            {/* Graph Area */}
            <div className="flex-1 min-h-[11rem] bg-white dark:bg-zinc-800 rounded-2xl border border-neutral-100 dark:border-zinc-700 p-4 shadow-sm relative">
                <div className="absolute top-3 left-4 text-[10px] font-bold text-neutral-400 dark:text-zinc-500 uppercase tracking-widest z-10">Real-time Telemetry</div>
                <ResponsiveContainer width="100%" height="100%">
                    <LineChart data={graphData}>
                        <XAxis dataKey="time" hide />
                        <YAxis domain={['auto', 'auto']} hide />
                        <Tooltip
                            contentStyle={{ backgroundColor: 'var(--glass-bg)', borderRadius: '16px', border: '1px solid var(--glass-border)', boxShadow: '0 10px 40px -10px rgba(0,0,0,0.1)', padding: '12px' }}
                            itemStyle={{ fontSize: '11px', color: 'var(--foreground)', fontWeight: 500, opacity: 0.7 }}
                            labelStyle={{ display: 'none' }}
                        />
                        {graphData.length > 0 && Object.keys(graphData[0]).filter(k => k !== 'time').map((key, index) => (
                            <Line
                                key={key}
                                type="monotone"
                                dataKey={key}
                                stroke={colors[index % colors.length]}
                                strokeWidth={2}
                                dot={false}
                                isAnimationActive={false}
                                strokeOpacity={0.8}
                            />
                        ))}
                    </LineChart>
                </ResponsiveContainer>
                {!isRunning && graphData.length === 0 && (
                    <div className="absolute inset-0 flex items-center justify-center">
                        <span className="text-neutral-300 dark:text-zinc-600 text-sm font-medium bg-neutral-50 dark:bg-zinc-900 px-4 py-2 rounded-full">Waiting for stream...</span>
                    </div>
                )}
            </div>

            {/* Torque / Motor Status */}
            {Object.keys(torqueData).length > 0 && (
                <div className="flex flex-col gap-2">
                    <span className="text-[10px] font-bold text-neutral-400 dark:text-zinc-500 uppercase tracking-widest px-1">Motor Load</span>
                    <div className="grid grid-cols-4 sm:grid-cols-6 md:grid-cols-8 gap-2">
                        {Object.entries(torqueData).map(([key, val]: [string, any]) => {
                            // Scale 0-1000
                            // < 500: Green, 500-800: Yellow, > 800: Red
                            const value = typeof val === 'number' ? val : 0;
                            const percentage = Math.min((value / 1000) * 100, 100);

                            let colorClass = "bg-green-500";
                            if (value > 800) colorClass = "bg-red-500";
                            else if (value > 500) colorClass = "bg-yellow-500";

                            // Shorten key name if needed
                            const label = key.replace(/link/g, 'L').replace(/_follower/g, ' (F)');

                            return (
                                <div key={key} className="bg-neutral-50 dark:bg-zinc-800 border border-neutral-100 dark:border-zinc-700 p-2 rounded-xl flex flex-col gap-1.5 shadow-sm">
                                    <div className="flex justify-between items-end">
                                        <span className="text-[10px] font-semibold text-neutral-600 dark:text-zinc-400 truncate" title={key}>{label}</span>
                                        <span className={`text-[10px] font-mono ${value > 500 ? 'text-neutral-900 dark:text-zinc-100 font-bold' : 'text-neutral-400 dark:text-zinc-500'}`}>{value}</span>
                                    </div>
                                    <div className="h-1.5 w-full bg-neutral-200 dark:bg-zinc-700 rounded-full overflow-hidden">
                                        <div
                                            className={`h-full rounded-full transition-all duration-300 ${colorClass}`}
                                            style={{ width: `${percentage}%` }}
                                        />
                                    </div>
                                </div>
                            )
                        })}
                    </div>
                </div>
            )}
        </>
    );
}
