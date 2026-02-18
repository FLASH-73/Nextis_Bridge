import React, { useState, useEffect, useRef, useCallback } from 'react';
import { Maximize2, Minimize2, AlertCircle, RefreshCw } from 'lucide-react';
import { camerasApi } from '../../lib/api';
import type { CameraStatusEntry } from '../../lib/api/types';

export interface CameraFeedProps {
  cameraId: string;
  mode?: 'contain' | 'cover' | 'fill';
  className?: string;
  aspectRatio?: string;
  maxStreamWidth?: number;
  quality?: number;
  showOverlay?: boolean;
  showFullscreenButton?: boolean;
  autoReconnect?: boolean;
  reconnectInterval?: number;
  label?: string;
  badge?: string;
  onFrameLoad?: () => void;
}

type FeedStatus = 'loading' | 'live' | 'error' | 'reconnecting';

const MAX_RETRIES = 5;

const objectFitClass: Record<string, string> = {
  contain: 'object-contain',
  cover: 'object-cover',
  fill: 'object-fill',
};

export default function CameraFeed({
  cameraId,
  mode = 'contain',
  className = '',
  aspectRatio = 'auto',
  maxStreamWidth = 800,
  quality = 80,
  showOverlay = true,
  showFullscreenButton = true,
  autoReconnect = true,
  reconnectInterval = 3000,
  label,
  badge,
  onFrameLoad,
}: CameraFeedProps) {
  const [status, setStatus] = useState<FeedStatus>('loading');
  const [retryCount, setRetryCount] = useState(0);
  const [countdown, setCountdown] = useState(0);
  const [resolution, setResolution] = useState<{ w: number; h: number } | null>(null);
  const [isFullscreen, setIsFullscreen] = useState(false);
  const [cacheBuster, setCacheBuster] = useState(Date.now());

  const containerRef = useRef<HTMLDivElement>(null);
  const countdownRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const retryTimeoutRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  // Build MJPEG src URL
  const src = `${camerasApi.videoFeedUrl(cameraId)}?max_width=${maxStreamWidth}&quality=${quality}&_t=${cacheBuster}`;

  // Fetch resolution once on mount
  useEffect(() => {
    let cancelled = false;
    camerasApi.status().then((statuses: Record<string, CameraStatusEntry>) => {
      if (cancelled) return;
      const entry = statuses[cameraId];
      if (entry?.actual_width && entry?.actual_height) {
        setResolution({ w: entry.actual_width, h: entry.actual_height });
      }
    }).catch(() => {});
    return () => { cancelled = true; };
  }, [cameraId]);

  // Cleanup timers on unmount
  useEffect(() => {
    return () => {
      if (countdownRef.current) clearInterval(countdownRef.current);
      if (retryTimeoutRef.current) clearTimeout(retryTimeoutRef.current);
    };
  }, []);

  // Listen for fullscreen changes
  useEffect(() => {
    const handler = () => {
      setIsFullscreen(!!document.fullscreenElement);
    };
    document.addEventListener('fullscreenchange', handler);
    return () => document.removeEventListener('fullscreenchange', handler);
  }, []);

  const startReconnect = useCallback(() => {
    if (!autoReconnect) return;

    setStatus('reconnecting');
    const seconds = Math.ceil(reconnectInterval / 1000);
    setCountdown(seconds);

    // Countdown timer
    if (countdownRef.current) clearInterval(countdownRef.current);
    let remaining = seconds;
    countdownRef.current = setInterval(() => {
      remaining -= 1;
      setCountdown(remaining);
      if (remaining <= 0 && countdownRef.current) {
        clearInterval(countdownRef.current);
        countdownRef.current = null;
      }
    }, 1000);

    // Schedule retry
    if (retryTimeoutRef.current) clearTimeout(retryTimeoutRef.current);
    retryTimeoutRef.current = setTimeout(() => {
      setCacheBuster(Date.now());
      setStatus('loading');
    }, reconnectInterval);
  }, [autoReconnect, reconnectInterval]);

  const handleLoad = useCallback(() => {
    setStatus('live');
    setRetryCount(0);
    onFrameLoad?.();
  }, [onFrameLoad]);

  const handleError = useCallback(() => {
    setRetryCount(prev => {
      const next = prev + 1;
      if (next >= MAX_RETRIES) {
        setStatus('error');
      } else {
        startReconnect();
      }
      return next;
    });
  }, [startReconnect]);

  const handleManualRetry = useCallback(() => {
    setRetryCount(0);
    setCacheBuster(Date.now());
    setStatus('loading');
  }, []);

  const toggleFullscreen = useCallback(() => {
    if (!containerRef.current) return;
    if (document.fullscreenElement) {
      document.exitFullscreen();
    } else {
      containerRef.current.requestFullscreen();
    }
  }, []);

  const displayLabel = label || cameraId;

  // Status dot color
  const dotColor =
    status === 'live' ? 'bg-green-500' :
    status === 'error' ? 'bg-red-500' :
    'bg-yellow-500 animate-pulse';

  const containerStyle: React.CSSProperties = {};
  if (aspectRatio !== 'auto') {
    containerStyle.aspectRatio = aspectRatio;
  }

  return (
    <div
      ref={containerRef}
      className={`w-full h-full bg-neutral-900 rounded-2xl overflow-hidden border border-white/5 relative group ${className}`}
      style={containerStyle}
    >
      {/* Loading shimmer */}
      {(status === 'loading' || status === 'reconnecting') && (
        <div className="absolute inset-0 bg-neutral-800 animate-pulse z-0" />
      )}

      {/* MJPEG stream */}
      {status !== 'error' && (
        <img
          src={src}
          alt={displayLabel}
          data-camera-id={cameraId}
          className={`w-full h-full ${objectFitClass[mode]} transition-opacity duration-300 ${status === 'live' ? 'opacity-100' : 'opacity-0'}`}
          onLoad={handleLoad}
          onError={handleError}
          draggable={false}
        />
      )}

      {/* Reconnecting overlay */}
      {status === 'reconnecting' && (
        <div className="absolute inset-0 flex items-center justify-center z-10">
          <div className="flex flex-col items-center gap-2">
            <RefreshCw className="w-6 h-6 text-white/30 animate-spin" />
            <span className="text-white/40 text-xs">Reconnecting in {countdown}s...</span>
          </div>
        </div>
      )}

      {/* Error state (after max retries) */}
      {status === 'error' && (
        <div className="absolute inset-0 flex items-center justify-center z-10">
          <div className="flex flex-col items-center gap-3">
            <AlertCircle className="w-8 h-8 text-white/20" />
            <span className="text-white/30 text-sm">Connection lost</span>
            <button
              onClick={handleManualRetry}
              className="px-3 py-1.5 bg-white/10 hover:bg-white/20 rounded-full text-xs text-white/70 transition-colors flex items-center gap-1.5"
            >
              <RefreshCw className="w-3 h-3" /> Retry
            </button>
          </div>
        </div>
      )}

      {/* Top-left: Label badge */}
      {showOverlay && (
        <div className="absolute top-3 left-3 z-20 pointer-events-none">
          <span className="bg-black/50 backdrop-blur-md rounded-full px-3 py-1 text-xs text-white border border-white/10 font-medium flex items-center gap-2">
            <div className={`w-2 h-2 rounded-full ${dotColor}`} />
            {displayLabel}
            {badge && (
              <span className="text-blue-400 text-[10px]">{badge}</span>
            )}
          </span>
        </div>
      )}

      {/* Bottom-right: Resolution badge */}
      {showOverlay && resolution && status === 'live' && (
        <div className="absolute bottom-3 right-3 z-20 pointer-events-none">
          <span className="bg-black/50 backdrop-blur-md rounded-full px-2.5 py-0.5 text-[10px] text-white/60 border border-white/5">
            {resolution.w}x{resolution.h}
          </span>
        </div>
      )}

      {/* Top-right: Fullscreen button */}
      {showFullscreenButton && status === 'live' && (
        <div className="absolute top-3 right-3 z-20 opacity-0 group-hover:opacity-100 transition-opacity">
          <button
            onClick={toggleFullscreen}
            className="bg-black/50 backdrop-blur-md rounded-full p-1.5 text-white/70 hover:text-white border border-white/10 transition-colors"
          >
            {isFullscreen ? <Minimize2 className="w-3.5 h-3.5" /> : <Maximize2 className="w-3.5 h-3.5" />}
          </button>
        </div>
      )}
    </div>
  );
}
