import React, { useRef, useCallback, useEffect } from 'react';
import { TURBO_COLORMAP, VIRIDIS_COLORMAP, applyColormap, ColormapType } from '../../../utils/colormaps';

/**
 * DepthVideoCanvas - Renders depth video with colormap applied via canvas
 */
export default function DepthVideoCanvas({
    videoUrl,
    videoRef,
    colormap,
    isPlaying,
    fromTimestamp,
}: {
    videoUrl: string;
    videoRef: (el: HTMLVideoElement | null) => void;
    colormap: ColormapType;
    isPlaying: boolean;
    fromTimestamp?: number;
}) {
    const canvasRef = useRef<HTMLCanvasElement>(null);
    const hiddenVideoRef = useRef<HTMLVideoElement>(null);
    const animationRef = useRef<number | null>(null);

    // Get the appropriate colormap LUT
    const getLUT = useCallback(() => {
        if (colormap === 'turbo') return TURBO_COLORMAP;
        if (colormap === 'viridis') return VIRIDIS_COLORMAP;
        return null; // grayscale
    }, [colormap]);

    // Render a single frame with colormap
    const renderFrame = useCallback(() => {
        const video = hiddenVideoRef.current;
        const canvas = canvasRef.current;
        if (!video || !canvas || video.readyState < 2) return;

        const ctx = canvas.getContext('2d');
        if (!ctx) return;

        // Match canvas size to video
        if (canvas.width !== video.videoWidth || canvas.height !== video.videoHeight) {
            canvas.width = video.videoWidth;
            canvas.height = video.videoHeight;
        }

        // Draw video frame
        ctx.drawImage(video, 0, 0);

        // Apply colormap if not grayscale
        const lut = getLUT();
        if (lut) {
            const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height);
            const data = imageData.data;

            for (let i = 0; i < data.length; i += 4) {
                const gray = data[i]; // R channel (grayscale video)
                const [r, g, b] = applyColormap(gray, lut);
                data[i] = r;
                data[i + 1] = g;
                data[i + 2] = b;
                // Alpha stays unchanged
            }

            ctx.putImageData(imageData, 0, 0);
        }
    }, [getLUT]);

    // Animation loop for playing video
    useEffect(() => {
        const video = hiddenVideoRef.current;
        if (!video) return;

        const animate = () => {
            if (!video.paused && !video.ended) {
                renderFrame();
                animationRef.current = requestAnimationFrame(animate);
            }
        };

        if (isPlaying) {
            // Actually play the hidden video and start animation loop
            video.play().then(() => {
                animate();
            }).catch(e => {
                console.warn('Depth video play failed:', e);
            });
        } else {
            // Pause the video and render current frame
            video.pause();
            renderFrame();
        }

        return () => {
            if (animationRef.current) {
                cancelAnimationFrame(animationRef.current);
            }
        };
    }, [isPlaying, renderFrame]);

    // Re-render when colormap changes
    useEffect(() => {
        renderFrame();
    }, [colormap, renderFrame]);

    // Handle video events
    useEffect(() => {
        const video = hiddenVideoRef.current;
        if (!video) return;

        const handleSeeked = () => renderFrame();
        const handleLoadedData = () => {
            // Seek to correct position for concatenated videos
            if (fromTimestamp && fromTimestamp > 0) {
                video.currentTime = fromTimestamp;
            }
            renderFrame();
        };

        video.addEventListener('seeked', handleSeeked);
        video.addEventListener('loadeddata', handleLoadedData);

        return () => {
            video.removeEventListener('seeked', handleSeeked);
            video.removeEventListener('loadeddata', handleLoadedData);
        };
    }, [renderFrame, fromTimestamp]);

    return (
        <div className="relative w-full h-full">
            <video
                ref={el => {
                    hiddenVideoRef.current = el;
                    videoRef(el);
                }}
                src={videoUrl}
                className="hidden"
                muted
                loop
                playsInline
                preload="auto"
                crossOrigin="anonymous"
            />
            <canvas
                ref={canvasRef}
                className="w-full h-full object-contain"
            />
        </div>
    );
}
