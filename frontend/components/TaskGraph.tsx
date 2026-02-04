import React, { useCallback } from 'react';
import {
    ReactFlow,
    MiniMap,
    Controls,
    Background,
    useNodesState,
    useEdgesState,
    addEdge,
    BackgroundVariant,
} from '@xyflow/react';
import '@xyflow/react/dist/style.css';

interface TaskGraphProps {
    plan: any[];
    darkMode?: boolean;
}

export default function TaskGraph({ plan, darkMode = false }: TaskGraphProps) {

    const getLabel = (task: any) => {
        if (task.task === 'move_to_bin' && task.params?.bin_id) return `Move -> Bin ${task.params.bin_id}`;
        if (task.task === 'pick_object' && task.params?.object_name) return `Pick: ${task.params.object_name}`;
        if (task.task === 'place_in_box' && task.params?.box_id) {
            const boxId = task.params.box_id;
            // If ID is generic, just say "Place -> Box"
            if (boxId.toLowerCase() === 'default' || boxId.toLowerCase() === 'box') {
                return 'Place -> Box';
            }
            if (boxId.toLowerCase().includes('box')) {
                return `Place -> ${boxId}`;
            }
            return `Place -> Box ${boxId}`;
        }
        if (task.task === 'home') return `Home`;
        return `${task.task}`;
    };

    // Theme-aware colors
    const nodeStyle = {
        background: darkMode ? '#18181B' : '#FFFFFF',
        color: darkMode ? '#FAFAFA' : '#1D1D1F',
        border: `1px solid ${darkMode ? '#3F3F46' : '#E5E5EA'}`,
        padding: '12px 20px',
        borderRadius: '100px', // Pill shape
        boxShadow: darkMode ? '0 4px 6px -1px rgba(0,0,0,0.3)' : '0 4px 6px -1px rgba(0,0,0,0.05)',
        fontSize: '12px',
        fontWeight: 500,
        width: 'auto',
        minWidth: 140,
        textAlign: 'center' as const,
        letterSpacing: '-0.01em'
    };

    const edgeColor = darkMode ? '#52525B' : '#D1D1D6';
    const bgColor = darkMode ? '#3F3F46' : '#E5E5EA';

    // Convert plan to nodes/edges
    const safePlan = Array.isArray(plan) ? plan : [];
    const initialNodes = safePlan.map((task, index) => ({
        id: `node-${index}`,
        position: { x: 100 + (index * 250), y: 100 },
        data: { label: getLabel(task) },
        type: 'default',
        style: nodeStyle
    }));

    const initialEdges = safePlan.slice(0, -1).map((_, index) => ({
        id: `edge-${index}`,
        source: `node-${index}`,
        target: `node-${index + 1}`,
        animated: true,
        style: { stroke: edgeColor, strokeWidth: 2 },
    }));

    const [nodes, setNodes, onNodesChange] = useNodesState(initialNodes);
    const [edges, setEdges, onEdgesChange] = useEdgesState(initialEdges);

    // Update nodes when plan changes
    React.useEffect(() => {
        const safePlan = Array.isArray(plan) ? plan : [];

        const newNodes = safePlan.map((task, index) => ({
            id: `node-${index}`,
            position: { x: 100 + (index * 250), y: 100 },
            data: { label: getLabel(task) },
            type: 'default',
            style: nodeStyle
        }));

        const newEdges = safePlan.slice(0, -1).map((_, index) => ({
            id: `edge-${index}`,
            source: `node-${index}`,
            target: `node-${index + 1}`,
            animated: true,
            style: { stroke: edgeColor, strokeWidth: 2 }
        }));

        setNodes(newNodes);
        setEdges(newEdges);
    }, [plan, darkMode, setNodes, setEdges]);

    const onConnect = useCallback(
        (params: any) => setEdges((eds) => addEdge(params, eds)),
        [setEdges],
    );

    return (
        <div style={{ width: '100%', height: '100%' }}>
            <ReactFlow
                nodes={nodes}
                edges={edges}
                onNodesChange={onNodesChange}
                onEdgesChange={onEdgesChange}
                onConnect={onConnect}
                colorMode={darkMode ? "dark" : "light"}
                fitView
                proOptions={{ hideAttribution: true }}
            >
                <Controls showInteractive={false} position="bottom-right" style={{ display: 'none' }} />
                {/* Hide controls for cleaner look, user can use scroll/pinch */}
                <Background variant={BackgroundVariant.Dots} gap={20} size={1} color={bgColor} />
            </ReactFlow>
        </div>
    );
}
