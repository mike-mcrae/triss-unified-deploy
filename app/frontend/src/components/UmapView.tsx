import { useEffect, useRef, useState } from "react";
import DeckGL from "@deck.gl/react";
import { ScatterplotLayer } from "@deck.gl/layers";
import { LinearInterpolator, OrthographicView } from "@deck.gl/core";
import type { PickingInfo } from "@deck.gl/core";

export interface MapPoint {
    n_id?: number;
    display_name?: string;
    school?: string;
    topic?: number;
    x: number;
    y: number;
    title?: string;
}

interface Props {
    points: MapPoint[];
    colorMode: "topic" | "school";
    onHoverPoint: (info: { x: number; y: number; object: MapPoint | null } | null) => void;
}

const TOPIC_COLORS: number[][] = [
    [99, 102, 241], [139, 92, 246], [236, 72, 153], [245, 158, 11],
    [16, 185, 129], [14, 165, 233], [217, 70, 239], [244, 63, 94]
];

const SCHOOL_COLORS: Record<string, number[]> = {
    "School of Engineering": [16, 185, 129],
    "School of Science": [99, 102, 241],
    "School of Humanities": [236, 72, 153],
    "School of Medicine": [245, 158, 11]
};

export default function UmapView({ points, colorMode, onHoverPoint }: Props) {
    const containerRef = useRef<HTMLDivElement | null>(null);
    const [size, setSize] = useState({ width: 800, height: 600 });
    const [viewState, setViewState] = useState({
        target: [0, 0, 0] as [number, number, number],
        zoom: 1,
        minZoom: -5,
        maxZoom: 10
    });

    useEffect(() => {
        if (!containerRef.current) return;
        const observer = new ResizeObserver((entries) => {
            const entry = entries[0];
            setSize({ width: entry.contentRect.width, height: entry.contentRect.height });
        });
        observer.observe(containerRef.current);
        return () => observer.disconnect();
    }, []);

    // Center based on points
    useEffect(() => {
        if (!points || points.length === 0 || !size.width) return;

        // Filter out any invalid points
        const validPoints = points.filter(p => typeof p.x === 'number' && typeof p.y === 'number');
        if (validPoints.length === 0) return;

        const xs = validPoints.map(p => p.x);
        const ys = validPoints.map(p => p.y);
        const minX = Math.min(...xs);
        const maxX = Math.max(...xs);
        const minY = Math.min(...ys);
        const maxY = Math.max(...ys);

        const centerX = (minX + maxX) / 2;
        const centerY = (minY + maxY) / 2;

        // Calculate appropriate zoom level based on data range
        const rangeX = maxX - minX;
        const rangeY = maxY - minY;
        const maxRange = Math.max(rangeX, rangeY, 1);
        const newZoom = Math.log2(Math.min(size.width, size.height) / maxRange) - 1;

        setViewState(prev => ({
            ...prev,
            target: [centerX, centerY, 0],
            zoom: isFinite(newZoom) ? newZoom : 1,
            transitionDuration: 1000,
            transitionInterpolator: new LinearInterpolator(["target", "zoom"])
        }));
    }, [points.length]);

    const layers = [
        new ScatterplotLayer<MapPoint>({
            id: "points-layer",
            data: points,
            pickable: true,
            opacity: 0.8,
            stroked: true,
            filled: true,
            radiusScale: 1,
            radiusMinPixels: 2,
            radiusMaxPixels: 10,
            lineWidthMinPixels: 1,
            getPosition: (d) => [d.x, d.y],
            getRadius: 15,
            getFillColor: (d) => {
                let color = [148, 163, 184];
                if (colorMode === "school" && d.school) {
                    color = SCHOOL_COLORS[d.school] || [148, 163, 184];
                } else {
                    color = TOPIC_COLORS[(d.topic || 0) % TOPIC_COLORS.length];
                }
                return color as [number, number, number];
            },
            getLineColor: [255, 255, 255],
            onHover: (info: PickingInfo<MapPoint>) => {
                if (info.object) {
                    onHoverPoint({ x: info.x, y: info.y, object: info.object });
                } else {
                    onHoverPoint(null);
                }
            }
        })
    ];

    return (
        <div style={{ width: "100%", height: "100%", position: "relative" }} ref={containerRef}>
            <DeckGL
                initialViewState={viewState}
                controller={true}
                layers={layers}
                views={new OrthographicView()}
                getCursor={({ isHovering }) => (isHovering ? "pointer" : "grab")}
            />
        </div>
    );
}
