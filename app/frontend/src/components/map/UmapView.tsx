import { useEffect, useMemo, useRef, useState } from "react";
import DeckGL from "@deck.gl/react";
import { ScatterplotLayer, TextLayer } from "@deck.gl/layers";
import { LinearInterpolator, OrthographicView } from "@deck.gl/core";
import type { PickingInfo } from "@deck.gl/core";
import type { PointRecord } from "../../pages/mapTypes";

type Props = {
  points: PointRecord[];
  filteredPoints: PointRecord[];
  colorMap: Record<string, number[]>;
  topicColorMap: Record<string, number[]>;
  policyColorMap: Record<string, number[]>;
  hoveredTopic: number | null;
  selectedTopic: number | null;
  selectedPolicy: number | null;
  showTopicNumbers: boolean;
  hideGreyedOutDots: boolean;
  colorMode: "topic" | "school" | "policy";
  fitSignal: number;
  onHoverPoint: (info: { x: number; y: number; object: PointRecord | null } | null) => void;
  onSelectPoint: (point: PointRecord | null) => void;
};

function getBounds(points: PointRecord[]) {
  if (!points.length) return null;
  let minX = points[0].x;
  let maxX = points[0].x;
  let minY = points[0].y;
  let maxY = points[0].y;
  points.forEach((p) => {
    minX = Math.min(minX, p.x);
    maxX = Math.max(maxX, p.x);
    minY = Math.min(minY, p.y);
    maxY = Math.max(maxY, p.y);
  });
  return { minX, maxX, minY, maxY };
}

export default function UmapView({
  points,
  filteredPoints,
  colorMap,
  topicColorMap,
  policyColorMap,
  hoveredTopic,
  selectedTopic,
  selectedPolicy,
  showTopicNumbers,
  hideGreyedOutDots,
  colorMode,
  fitSignal,
  onHoverPoint,
  onSelectPoint
}: Props) {
  const containerRef = useRef<HTMLDivElement | null>(null);
  const [size, setSize] = useState({ width: 800, height: 600 });
  const [viewState, setViewState] = useState({
    target: [0, 0, 0] as [number, number, number],
    zoom: 0,
    minZoom: -8,
    maxZoom: 12
  });
  const pointsRef = useRef<PointRecord[]>(points);
  const filteredRef = useRef<PointRecord[]>(filteredPoints);
  const filteredKeys = useMemo(() => {
    const keys = new Set<string>();
    filteredPoints.forEach((point) => keys.add(point.key));
    return keys;
  }, [filteredPoints]);

  useEffect(() => {
    pointsRef.current = points;
  }, [points]);

  useEffect(() => {
    filteredRef.current = filteredPoints;
  }, [filteredPoints]);

  const centroidData = useMemo(() => {
    const source = filteredPoints.length ? filteredPoints : points;
    const accum = new Map<number, { x: number; y: number; count: number }>();
    source.forEach((point) => {
      if (point.topic === null || point.topic === undefined) return;
      const entry = accum.get(point.topic) || { x: 0, y: 0, count: 0 };
      entry.x += point.x;
      entry.y += point.y;
      entry.count += 1;
      accum.set(point.topic, entry);
    });
    return Array.from(accum.entries()).map(([topic, entry]) => ({
      topic,
      x: entry.x / entry.count,
      y: entry.y / entry.count,
      count: entry.count
    }));
  }, [filteredPoints, points]);

  useEffect(() => {
    if (!containerRef.current) return;
    const observer = new ResizeObserver((entries) => {
      const entry = entries[0];
      setSize({ width: entry.contentRect.width, height: entry.contentRect.height });
    });
    observer.observe(containerRef.current);
    return () => observer.disconnect();
  }, []);

  useEffect(() => {
    if (!size.width || !size.height) return;
    const source = filteredRef.current.length ? filteredRef.current : pointsRef.current;
    const bounds = getBounds(source);
    if (!bounds) return;

    const padding = 140;
    const rangeX = Math.max(bounds.maxX - bounds.minX, 1e-6);
    const rangeY = Math.max(bounds.maxY - bounds.minY, 1e-6);
    const range = Math.max(rangeX, rangeY);
    const available = Math.min(size.width, size.height) - padding;
    if (available <= 0) return;

    const scale = available / range;
    const zoom = Math.log2(scale);
    const target: [number, number, number] = [
      (bounds.minX + bounds.maxX) / 2,
      (bounds.minY + bounds.maxY) / 2,
      0
    ];

    setViewState((prev) => ({
      ...prev,
      target,
      zoom: Math.max(-6, Math.min(10, zoom)),
      transitionDuration: 800,
      transitionInterpolator: new LinearInterpolator(["target", "zoom"])
    }));
  }, [size, fitSignal]);

  const layer = useMemo(() => {
    const data = hideGreyedOutDots ? filteredPoints : points;
    return new ScatterplotLayer<PointRecord>({
      id: "triss-points",
      data,
      pickable: true,
      radiusMinPixels: 0.48,
      radiusMaxPixels: 2.16,
      getPosition: (d) => [d.x, d.y],
      getRadius: (d) => {
        const similarity = typeof d.centroid_similarity === "number" ? d.centroid_similarity : 0.5;
        return 1.2 + similarity * 2.64;
      },
      getFillColor: (d) => {
        const isActive = filteredKeys.has(d.key);
        if (!isActive) {
          return [176, 182, 190, 44] as [number, number, number, number];
        }
        if (colorMode === "school") {
          const base = d.school ? colorMap[d.school] || [80, 80, 90] : [80, 80, 90];
          return [base[0], base[1], base[2], 190] as [number, number, number, number];
        }
        if (colorMode === "policy") {
          const base = d.policy_domain != null ? policyColorMap[String(d.policy_domain)] || [80, 80, 90] : [80, 80, 90];
          const isSelected = selectedPolicy !== null && d.policy_domain !== selectedPolicy;
          if (isSelected) {
            return [200, 200, 200, 40] as [number, number, number, number];
          }
          return [base[0], base[1], base[2], 190] as [number, number, number, number];
        }
        const base = d.topic != null ? colorMap[String(d.topic)] || [60, 90, 120] : [80, 80, 90];
        const isTopicFocus = hoveredTopic !== null && d.topic !== hoveredTopic;
        const isSelected = selectedTopic !== null && d.topic !== selectedTopic;
        if (isTopicFocus || isSelected) {
          return [200, 200, 200, 40] as [number, number, number, number];
        }
        return [base[0], base[1], base[2], 190] as [number, number, number, number];
      },
      updateTriggers: {
        getFillColor: [colorMode, colorMap, hoveredTopic, selectedTopic, selectedPolicy, filteredKeys, policyColorMap]
      },
      onHover: (info: PickingInfo<PointRecord>) => {
        if (
          info.object &&
          filteredKeys.has(info.object.key) &&
          typeof info.x === "number" &&
          typeof info.y === "number"
        ) {
          onHoverPoint({ x: info.x, y: info.y, object: info.object });
        } else {
          onHoverPoint(null);
        }
      },
      onClick: (info: PickingInfo<PointRecord>) => {
        if (info.object && filteredKeys.has(info.object.key)) {
          onSelectPoint(info.object);
          return;
        }
        onSelectPoint(null);
      }
    });
  }, [points, filteredPoints, hideGreyedOutDots, filteredKeys, colorMap, hoveredTopic, selectedTopic, selectedPolicy, colorMode, onHoverPoint, onSelectPoint, policyColorMap]);

  const labelLayer = useMemo(() => {
    return new TextLayer({
      id: "triss-centroids",
      data: centroidData,
      getPosition: (d: { x: number; y: number }) => [d.x, d.y],
      getText: (d: { topic: number }) => String(d.topic),
      getSize: 16,
      sizeUnits: "pixels",
      getColor: [0, 0, 0, 220],
      background: false,
      pickable: false
    });
  }, [centroidData]);

  const labelCircleLayer = useMemo(() => {
    return new ScatterplotLayer({
      id: "triss-centroid-circles",
      data: centroidData,
      getPosition: (d: { x: number; y: number }) => [d.x, d.y],
      radiusUnits: "pixels",
      getRadius: 12,
      filled: true,
      stroked: false,
      getFillColor: (d: { topic: number }) => {
        const base = topicColorMap[String(d.topic)] || [120, 120, 120];
        return [base[0], base[1], base[2], 128] as [number, number, number, number];
      },
      pickable: false
    });
  }, [centroidData, topicColorMap]);

  return (
    <div className="atlas-map-surface" ref={containerRef}>
      <DeckGL
        width={size.width}
        height={size.height}
        viewState={viewState}
        views={new OrthographicView()}
        controller
        layers={showTopicNumbers ? [layer, labelCircleLayer, labelLayer] : [layer]}
        onViewStateChange={({ viewState: next }) => setViewState(next as typeof viewState)}
        getCursor={({ isHovering }) => (isHovering ? "pointer" : "grab")}
      />
    </div>
  );
}
