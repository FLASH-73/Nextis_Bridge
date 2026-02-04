import { useState, useEffect, useRef } from 'react';

interface Size {
    width: number;
    height: number;
}

interface UseResizableProps {
    initialSize: Size;
    minSize?: Size;
    maxSize?: Size;
}

export function useResizable({ initialSize, minSize = { width: 300, height: 200 } }: UseResizableProps) {
    const [size, setSize] = useState<Size>(initialSize);
    const [isResizing, setIsResizing] = useState(false);

    // We don't strictly need a ref for the element itself unless we want to measure it initially
    // But we do need to track the start position of the mouse ensuring smooth delta updates
    const startPos = useRef({ x: 0, y: 0 });
    const startSize = useRef(initialSize);

    useEffect(() => {
        const handleMouseMove = (e: MouseEvent) => {
            if (!isResizing) return;

            // Calculate delta
            const dx = e.clientX - startPos.current.x;
            const dy = e.clientY - startPos.current.y;

            // Apply new size
            const newWidth = Math.max(minSize.width, startSize.current.width + dx);
            const newHeight = Math.max(minSize.height, startSize.current.height + dy);

            setSize({
                width: newWidth,
                height: newHeight
            });
        };

        const handleMouseUp = () => {
            if (isResizing) {
                setIsResizing(false);
                document.body.style.cursor = 'default';
                document.body.style.userSelect = 'auto'; // Re-enable selection
            }
        };

        if (isResizing) {
            document.addEventListener('mousemove', handleMouseMove);
            document.addEventListener('mouseup', handleMouseUp);
            document.body.style.cursor = 'nwse-resize'; // Force cursor global
            document.body.style.userSelect = 'none';    // Prevent text selection
        }

        return () => {
            document.removeEventListener('mousemove', handleMouseMove);
            document.removeEventListener('mouseup', handleMouseUp);
            // Cleanup styles if unmounted mid-drag
            if (isResizing) {
                document.body.style.cursor = 'default';
                document.body.style.userSelect = 'auto';
            }
        };
    }, [isResizing, minSize]);

    const handleResizeMouseDown = (e: React.MouseEvent) => {
        // Prevent conflict with draggable parent if possible, stop propagation
        e.stopPropagation();

        setIsResizing(true);
        startPos.current = { x: e.clientX, y: e.clientY };
        startSize.current = size;
    };

    return { size, isResizing, handleResizeMouseDown, setSize };
}
