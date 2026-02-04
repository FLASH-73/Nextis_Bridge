import { useState, useEffect, useRef } from 'react';

export function useDraggable(initialPosition = { x: 0, y: 0 }) {
    // State for consumers who need it (syncs strictly on Drag End)
    const [position, setPosition] = useState(initialPosition);
    const [isDragging, setIsDragging] = useState(false);

    // Refs for High-Perf mutable physics
    const currentPos = useRef(initialPosition);
    const ref = useRef<HTMLDivElement>(null);
    const dragStartMouse = useRef({ x: 0, y: 0 });
    const dragStartPos = useRef({ x: 0, y: 0 });

    // Apply Transform Helper
    const updateTransform = () => {
        if (ref.current) {
            ref.current.style.transform = `translate(${currentPos.current.x}px, ${currentPos.current.y}px)`;
        }
    };

    // Initialize position on mount
    useEffect(() => {
        currentPos.current = initialPosition;
        updateTransform();
    }, []);

    useEffect(() => {
        const handleMouseMove = (e: MouseEvent) => {
            if (!isDragging) return;

            const dx = e.clientX - dragStartMouse.current.x;
            const dy = e.clientY - dragStartMouse.current.y;

            currentPos.current = {
                x: dragStartPos.current.x + dx,
                y: dragStartPos.current.y + dy
            };

            // Direct DOM update (No React Re-render)
            updateTransform();
        };

        const handleMouseUp = () => {
            if (isDragging) {
                setIsDragging(false);
                setPosition(currentPos.current); // Sync state at end
                if (ref.current) ref.current.style.willChange = 'auto';
            }
        };

        if (isDragging) {
            document.addEventListener('mousemove', handleMouseMove);
            document.addEventListener('mouseup', handleMouseUp);
        }

        return () => {
            document.removeEventListener('mousemove', handleMouseMove);
            document.removeEventListener('mouseup', handleMouseUp);
        };
    }, [isDragging]);

    const handleMouseDown = (e: React.MouseEvent) => {
        setIsDragging(true);
        dragStartMouse.current = { x: e.clientX, y: e.clientY };
        dragStartPos.current = { ...currentPos.current };
        if (ref.current) ref.current.style.willChange = 'transform';
    };

    return { position, isDragging, handleMouseDown, ref };
}
