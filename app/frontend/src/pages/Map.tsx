import { useEffect, useMemo, useRef, useState } from "react";
import { fetchMapFilters, fetchMapPoints, fetchMapResearchers, fetchMapTopics } from "../api";
import FiltersPanel from "../components/map/FiltersPanel";
import TopicLegend from "../components/map/TopicLegend";
import Tooltip from "../components/map/Tooltip";
import UmapView from "../components/map/UmapView";
import type { FiltersData, HoverInfo, PointRecord, TopicsMap } from "./mapTypes";
import "../styles/map.css";

const COLOR_PALETTE: number[][] = [
  [32, 78, 128],
  [196, 61, 28],
  [22, 148, 106],
  [235, 170, 40],
  [143, 52, 168],
  [38, 168, 197],
  [204, 96, 125],
  [83, 68, 158],
  [143, 167, 37],
  [220, 108, 29]
];

const SCHOOL_PALETTE: number[][] = [
  [18, 113, 148],
  [208, 89, 45],
  [56, 150, 92],
  [176, 88, 178],
  [98, 92, 200],
  [186, 142, 52],
  [45, 145, 177],
  [170, 76, 110],
  [110, 136, 40],
  [210, 130, 42]
];

const OMIT_TOPICS = new Set<number>([]); // No outlier clusters in current K=25 pipeline

function buildColorMap(topics: FiltersData["topics"]): Record<string, number[]> {
  const ids = topics.map((t) => t.topic).sort((a, b) => a - b);
  const map: Record<string, number[]> = {};
  ids.forEach((id, index) => {
    map[String(id)] = COLOR_PALETTE[index % COLOR_PALETTE.length];
  });
  return map;
}

function filterPoints(
  points: PointRecord[],
  school: string,
  department: string,
  researcherId: number | null,
  topic: number | null,
  policy: number | null
) {
  return points.filter((point) => {
    if (point.topic !== null && point.topic !== undefined && OMIT_TOPICS.has(point.topic)) {
      return false;
    }
    if (school !== "All" && point.school !== school) return false;
    if (department && point.department !== department) return false;
    if (researcherId !== null && point.n_id !== researcherId) return false;
    if (topic !== null && point.topic !== topic) return false;
    if (policy !== null && point.policy_domain !== policy) return false;
    return true;
  });
}

export default function MapPage() {
  const [publicationPoints, setPublicationPoints] = useState<PointRecord[]>([]);
  const [researcherPoints, setResearcherPoints] = useState<PointRecord[]>([]);
  const [topics, setTopics] = useState<TopicsMap>({});
  const [filtersData, setFiltersData] = useState<FiltersData | null>(null);
  const [hoverInfo, setHoverInfo] = useState<HoverInfo | null>(null);
  const [loadingPoints, setLoadingPoints] = useState(true);
  const [loadingFilters, setLoadingFilters] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const [selectedSchool, setSelectedSchool] = useState("All");
  const [selectedDepartment, setSelectedDepartment] = useState("");
  const [selectedResearcher, setSelectedResearcher] = useState<number | null>(null);
  const [selectedTopic, setSelectedTopic] = useState<number | null>(null);
  const [selectedPolicy, setSelectedPolicy] = useState<number | null>(null);
  const [hoveredTopic, setHoveredTopic] = useState<number | null>(null);
  const [fitSignal, setFitSignal] = useState(0);
  const fitInitialized = useRef(false);
  const [selectedPoint, setSelectedPoint] = useState<PointRecord | null>(null);
  const [dataMode, setDataMode] = useState<"publication" | "researcher">("publication");
  const [showTopicNumbers, setShowTopicNumbers] = useState(true);
  const [hideGreyedOutDots, setHideGreyedOutDots] = useState(false);
  const [colorMode, setColorMode] = useState<"topic" | "school" | "policy">("topic");
  const [showHoverCard, setShowHoverCard] = useState(true);

  useEffect(() => {
    fetchMapPoints()
      .then(setPublicationPoints)
      .catch((err) => setError(err.message));

    fetchMapResearchers()
      .then(setResearcherPoints)
      .catch((err) => setError(err.message))
      .finally(() => setLoadingPoints(false));

    fetchMapTopics()
      .then(setTopics)
      .catch((err) => setError(err.message));

    fetchMapFilters()
      .then((data) => {
        setFiltersData(data);
        setSelectedSchool(data.schools[0] ?? "All");
      })
      .catch((err) => setError(err.message))
      .finally(() => setLoadingFilters(false));
  }, []);

  const points = dataMode === "publication" ? publicationPoints : researcherPoints;

  const filteredPoints = useMemo(() => {
    return filterPoints(points, selectedSchool, selectedDepartment, selectedResearcher, selectedTopic, selectedPolicy);
  }, [points, selectedSchool, selectedDepartment, selectedResearcher, selectedTopic, selectedPolicy]);

  const displayPoints = useMemo(() => {
    return points.filter((point) => !(point.topic !== null && point.topic !== undefined && OMIT_TOPICS.has(point.topic)));
  }, [points]);

  const visibleTopics = useMemo(() => {
    if (!filtersData) return [];
    return filtersData.topics.filter((topic) => !OMIT_TOPICS.has(topic.topic));
  }, [filtersData]);

  const topicNameMap = useMemo(() => {
    const map = new globalThis.Map<number, string>();
    visibleTopics.forEach((topic) => {
      map.set(topic.topic, topic.topic_name || `Topic ${topic.topic}`);
    });
    Object.values(topics).forEach((topic) => {
      map.set(topic.topic_id, topic.topic_name || `Topic ${topic.topic_id}`);
    });
    return map;
  }, [visibleTopics, topics]);

  const topicLegendItems = useMemo(() => {
    const counts = new globalThis.Map<number, number>();
    filteredPoints.forEach((point) => {
      if (point.topic === null || point.topic === undefined) return;
      if (OMIT_TOPICS.has(point.topic)) return;
      counts.set(point.topic, (counts.get(point.topic) || 0) + 1);
    });
    return Array.from(counts.entries())
      .map(([topic, count]) => ({
        key: String(topic),
        label: `${topic}. ${topicNameMap.get(topic) || ""}`.trim(),
        count
      }))
      .sort((a, b) => Number(a.key) - Number(b.key));
  }, [filteredPoints, topicNameMap]);

  const schoolLegendItems = useMemo(() => {
    if (!filtersData) return [];
    const counts = new globalThis.Map<string, number>();
    filteredPoints.forEach((point) => {
      if (!point.school) return;
      counts.set(point.school, (counts.get(point.school) || 0) + 1);
    });
    return filtersData.schools
      .filter((school) => school !== "All")
      .map((school) => ({
        key: school,
        label: school,
        count: counts.get(school) || 0
      }));
  }, [filteredPoints, filtersData]);

  const policyLegendItems = useMemo(() => {
    if (!filtersData) return [];
    const counts = new globalThis.Map<number, number>();
    filteredPoints.forEach((point) => {
      if (point.policy_domain === null || point.policy_domain === undefined) return;
      counts.set(point.policy_domain, (counts.get(point.policy_domain) || 0) + 1);
    });
    return filtersData.policies.map((policy) => ({
      key: String(policy.policy_domain),
      label: policy.policy_name || `Policy ${policy.policy_domain}`,
      count: counts.get(policy.policy_domain) || 0,
    }));
  }, [filteredPoints, filtersData]);

  const topicColorMap = useMemo(() => {
    if (!filtersData) return {};
    return buildColorMap(visibleTopics);
  }, [filtersData, visibleTopics]);

  const schoolColorMap = useMemo(() => {
    if (!filtersData) return {};
    const schools = filtersData.schools.filter((school) => school !== "All");
    const map: Record<string, number[]> = {};
    schools.forEach((school, index) => {
      map[school] = SCHOOL_PALETTE[index % SCHOOL_PALETTE.length];
    });
    return map;
  }, [filtersData]);

  const policyColorMap = useMemo(() => {
    if (!filtersData) return {};
    const policies = filtersData.policies;
    const map: Record<string, number[]> = {};
    policies.forEach((policy, index) => {
      map[String(policy.policy_domain)] = COLOR_PALETTE[index % COLOR_PALETTE.length];
    });
    return map;
  }, [filtersData]);

  const colorMap = colorMode === "topic" ? topicColorMap : colorMode === "school" ? schoolColorMap : policyColorMap;
  const selectedTopicDetails =
    selectedTopic !== null && selectedTopic !== undefined ? topics[String(selectedTopic)] : undefined;

  const onReset = () => {
    setSelectedSchool("All");
    setSelectedDepartment("");
    setSelectedResearcher(null);
    setSelectedTopic(null);
    setSelectedPolicy(null);
    setHoveredTopic(null);
    setSelectedPoint(null);
    setHideGreyedOutDots(false);
  };

  const onSelectDataMode = (mode: "publication" | "researcher") => {
    setDataMode(mode);
    setSelectedResearcher(null);
    setSelectedDepartment("");
    setSelectedTopic(null);
    setSelectedPolicy(null);
    setHoveredTopic(null);
    setSelectedPoint(null);
  };

  const onSelectSchool = (school: string) => {
    setSelectedSchool(school);
    setSelectedDepartment("");
  };

  const onSelectDepartment = (department: string) => {
    setSelectedDepartment(department);
  };

  const onSelectResearcher = (nId: number | null) => {
    setSelectedResearcher(nId);
  };

  const onSelectTopic = (topicId: number | null) => {
    setSelectedTopic(topicId);
  };

  const onSelectPolicy = (policyId: number | null) => {
    setSelectedPolicy(policyId);
  };

  const onSelectColorMode = (mode: "topic" | "school" | "policy") => {
    setColorMode(mode);
    if (mode !== "topic") {
      setSelectedTopic(null);
      setHoveredTopic(null);
    }
    if (mode !== "policy") {
      setSelectedPolicy(null);
    }
  };

  const onHoverPoint = (info: HoverInfo | null) => {
    setHoverInfo(info);
  };

  const onSelectPoint = (point: PointRecord | null) => {
    setSelectedPoint(point);
  };

  useEffect(() => {
    if (!points.length) return;
    if (!fitInitialized.current) {
      fitInitialized.current = true;
    }
    setFitSignal((value) => value + 1);
  }, [points.length, selectedSchool, selectedDepartment, selectedResearcher, selectedTopic, selectedPolicy]);

  useEffect(() => {
    if (selectedTopic !== null && OMIT_TOPICS.has(selectedTopic)) {
      setSelectedTopic(null);
    }
  }, [selectedTopic]);

  if (!filtersData) {
    return (
      <div className="page-container">
        <div className="loading-state">
          {loadingFilters ? "Loading TRISS atlas..." : error || "Failed to load filters."}
        </div>
      </div>
    );
  }

  return (
    <div className="page-container fade-in">
      <div className="atlas-shell">
        <header className="atlas-header">
          <div>
            <p className="atlas-eyebrow">TRISS Research Atlas</p>
            <h1 className="page-title">Topic Cartography</h1>
            <p className="atlas-subtitle">
              Explore publications or researchers across schools, departments, and policy/topic layers. Hover to reveal metadata, click themes to isolate patterns.
            </p>
          </div>
          <div className="atlas-counts">
            <div>Showing</div>
            <div className="atlas-count-number">{filteredPoints.length}</div>
            <div className="atlas-count-total">of {displayPoints.length}</div>
          </div>
        </header>

        <main className="atlas-main">
          <section className="atlas-filters">
            <FiltersPanel
              dataMode={dataMode}
              schools={filtersData.schools}
              departmentsBySchool={filtersData.departments_by_school}
              researchers={filtersData.researchers}
              topics={visibleTopics}
              policies={filtersData.policies}
              selectedSchool={selectedSchool}
              selectedDepartment={selectedDepartment}
              selectedResearcher={selectedResearcher}
              selectedTopic={selectedTopic}
              selectedPolicy={selectedPolicy}
              showTopicNumbers={showTopicNumbers}
              hideGreyedOutDots={hideGreyedOutDots}
              colorMode={colorMode}
              showHoverCard={showHoverCard}
              onSelectDataMode={onSelectDataMode}
              onSelectSchool={onSelectSchool}
              onSelectDepartment={onSelectDepartment}
              onSelectResearcher={onSelectResearcher}
              onSelectTopic={onSelectTopic}
              onSelectPolicy={onSelectPolicy}
              onToggleTopicNumbers={setShowTopicNumbers}
              onToggleHideGreyedOutDots={setHideGreyedOutDots}
              onSelectColorMode={onSelectColorMode}
              onToggleHoverCard={setShowHoverCard}
              onReset={onReset}
            />
          </section>

          <section className="atlas-map-panel">
            <div className="atlas-map-card">
              {displayPoints.length ? (
                <UmapView
                  points={displayPoints}
                  filteredPoints={filteredPoints}
                  colorMap={colorMap}
                  topicColorMap={topicColorMap}
                  policyColorMap={policyColorMap}
                  hoveredTopic={hoveredTopic}
                  selectedTopic={selectedTopic}
                  selectedPolicy={selectedPolicy}
                  showTopicNumbers={showTopicNumbers}
                  hideGreyedOutDots={hideGreyedOutDots}
                  colorMode={colorMode}
                  fitSignal={fitSignal}
                  onHoverPoint={onHoverPoint}
                  onSelectPoint={onSelectPoint}
                />
              ) : null}

              {loadingPoints || error || !displayPoints.length ? (
                <div className="atlas-map-status">
                  <div className="atlas-map-status-title">
                    {loadingPoints ? "Loading map data..." : "Map data unavailable"}
                  </div>
                  <p>
                    {error ? `API error: ${error}` : "Check that the backend is running at http://localhost:8000."}
                  </p>
                </div>
              ) : null}

              <button className="atlas-reset" type="button" onClick={() => setFitSignal((v) => v + 1)}>
                Reset view
              </button>
            </div>

            <TopicLegend
              title={colorMode === "topic" ? "Topics" : colorMode === "school" ? "Schools" : "Policy Domains"}
              items={colorMode === "topic" ? topicLegendItems : colorMode === "school" ? schoolLegendItems : policyLegendItems}
              colorMap={colorMode === "topic" ? topicColorMap : colorMode === "school" ? schoolColorMap : policyColorMap}
              activeKey={
                colorMode === "topic"
                  ? selectedTopic !== null ? String(selectedTopic) : null
                  : colorMode === "school"
                    ? (selectedSchool !== "All" ? selectedSchool : null)
                    : (selectedPolicy !== null ? String(selectedPolicy) : null)
              }
              onHoverItem={(key) => {
                if (colorMode === "topic") {
                  setHoveredTopic(key ? Number(key) : null);
                }
              }}
              onClickItem={(key) => {
                if (!key) return;
                if (colorMode === "topic") {
                  onSelectTopic(Number(key));
                } else if (colorMode === "school") {
                  onSelectSchool(key);
                } else {
                  onSelectPolicy(Number(key));
                }
              }}
              onClear={() => {
                if (colorMode === "topic") {
                  onSelectTopic(null);
                } else if (colorMode === "school") {
                  onSelectSchool("All");
                } else {
                  onSelectPolicy(null);
                }
              }}
            />

            {selectedTopic !== null ? (
              <section className="atlas-topic-card">
                <div className="atlas-topic-card-header">
                  <h3>
                    Topic {selectedTopic}:{" "}
                    {selectedTopicDetails?.topic_name || topicNameMap.get(selectedTopic) || "Unnamed"}
                  </h3>
                  <button type="button" className="atlas-ghost atlas-small" onClick={() => onSelectTopic(null)}>
                    Clear
                  </button>
                </div>
                <p>
                  {selectedTopicDetails?.topic_description ||
                    "No topic description available for this cluster in the current analysis file."}
                </p>
                {selectedTopicDetails?.top_words?.length ? (
                  <div className="atlas-topic-words">
                    {selectedTopicDetails.top_words.map((word) => (
                      <span key={word}>{word}</span>
                    ))}
                  </div>
                ) : null}
              </section>
            ) : null}
          </section>
        </main>

        {showHoverCard && hoverInfo?.object ? <Tooltip info={{ x: hoverInfo.x, y: hoverInfo.y, object: hoverInfo.object }} /> : null}

        {selectedPoint ? (
          <aside className="atlas-detail-card">
            <div className="atlas-detail-header">
              <h3>{selectedPoint.title || (dataMode === "researcher" ? "Researcher" : "Untitled abstract")}</h3>
              <button type="button" className="atlas-ghost atlas-small" onClick={() => setSelectedPoint(null)}>
                Close
              </button>
            </div>
            <div className="atlas-detail-meta">
              <div>{selectedPoint.name || "Unknown researcher"}</div>
              <div>
                {selectedPoint.school || ""}
                {selectedPoint.department ? ` / ${selectedPoint.department}` : ""}
              </div>
              {selectedPoint.topic !== null && selectedPoint.topic !== undefined ? (
                <button type="button" onClick={() => onSelectTopic(selectedPoint.topic ?? null)}>
                  Topic {selectedPoint.topic}: {selectedPoint.topic_name || "Unnamed"}
                </button>
              ) : (
                <span>No topic assigned</span>
              )}
            </div>
            {dataMode === "publication" && selectedPoint.abstract ? (
              <p className="atlas-detail-abstract">{selectedPoint.abstract}</p>
            ) : null}
          </aside>
        ) : null}
      </div>
    </div>
  );
}
