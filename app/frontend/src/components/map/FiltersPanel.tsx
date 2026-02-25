import { useEffect, useMemo, useState } from "react";
import type { FiltersData } from "../../pages/mapTypes";

type Props = {
  schools: FiltersData["schools"];
  departmentsBySchool: FiltersData["departments_by_school"];
  researchers: FiltersData["researchers"];
  topics: FiltersData["topics"];
  policies: FiltersData["policies"];
  dataMode: "publication" | "researcher";
  selectedSchool: string;
  selectedDepartment: string;
  selectedResearcher: number | null;
  selectedTopic: number | null;
  selectedPolicy: number | null;
  showTopicNumbers: boolean;
  hideGreyedOutDots: boolean;
  colorMode: "topic" | "school" | "policy";
  showHoverCard: boolean;
  onSelectDataMode: (value: "publication" | "researcher") => void;
  onSelectSchool: (value: string) => void;
  onSelectDepartment: (value: string) => void;
  onSelectResearcher: (value: number | null) => void;
  onSelectTopic: (value: number | null) => void;
  onSelectPolicy: (value: number | null) => void;
  onToggleTopicNumbers: (value: boolean) => void;
  onToggleHideGreyedOutDots: (value: boolean) => void;
  onSelectColorMode: (value: "topic" | "school" | "policy") => void;
  onToggleHoverCard: (value: boolean) => void;
  onReset: () => void;
};

export default function FiltersPanel({
  schools,
  departmentsBySchool,
  researchers,
  topics,
  policies,
  dataMode,
  selectedSchool,
  selectedDepartment,
  selectedResearcher,
  selectedTopic,
  selectedPolicy,
  showTopicNumbers,
  hideGreyedOutDots,
  colorMode,
  showHoverCard,
  onSelectDataMode,
  onSelectSchool,
  onSelectDepartment,
  onSelectResearcher,
  onSelectTopic,
  onSelectPolicy,
  onToggleTopicNumbers,
  onToggleHideGreyedOutDots,
  onSelectColorMode,
  onToggleHoverCard,
  onReset
}: Props) {
  const [researcherQuery, setResearcherQuery] = useState("");

  const departments = useMemo(() => {
    if (!selectedSchool || selectedSchool === "All") return [];
    return departmentsBySchool[selectedSchool] || [];
  }, [selectedSchool, departmentsBySchool]);

  const matches = useMemo(() => {
    const query = researcherQuery.trim().toLowerCase();
    if (query.length < 2) return [];
    return researchers.filter((researcher) => researcher.name.toLowerCase().includes(query)).slice(0, 6);
  }, [researcherQuery, researchers]);

  useEffect(() => {
    if (selectedResearcher === null) {
      setResearcherQuery("");
      return;
    }
    const match = researchers.find((r) => r.n_id === selectedResearcher);
    if (match) {
      setResearcherQuery(match.name);
    }
  }, [selectedResearcher, researchers]);

  const onResearcherInput = (value: string) => {
    setResearcherQuery(value);
    const match = researchers.find((r) => r.name.toLowerCase() === value.toLowerCase());
    if (match) {
      onSelectResearcher(match.n_id);
    } else if (value.trim() === "") {
      onSelectResearcher(null);
    }
  };

  return (
    <div className="atlas-panel">
      <div className="atlas-panel-header">
        <h2>Filters</h2>
        <button type="button" className="atlas-ghost" onClick={onReset}>
          Reset
        </button>
      </div>

      <label className="atlas-field">
        <span>School</span>
        <select value={selectedSchool} onChange={(e) => onSelectSchool(e.target.value)}>
          {schools.map((school) => (
            <option key={school} value={school}>
              {school}
            </option>
          ))}
        </select>
      </label>

      <label className="atlas-field">
        <span>Department</span>
        <select
          value={selectedDepartment}
          onChange={(e) => onSelectDepartment(e.target.value)}
          disabled={selectedSchool === "All" || departments.length === 0}
        >
          <option value="">All departments</option>
          {departments.map((department) => (
            <option key={department} value={department}>
              {department}
            </option>
          ))}
        </select>
      </label>

      <label className="atlas-field">
        <span>Researcher</span>
        <input
          list="atlas-researcher-options"
          placeholder="Type a name"
          value={researcherQuery}
          onChange={(e) => onResearcherInput(e.target.value)}
        />
        <datalist id="atlas-researcher-options">
          {researchers.map((researcher) => (
            <option key={researcher.n_id} value={researcher.name} />
          ))}
        </datalist>
        {matches.length ? (
          <div className="atlas-match-list">
            {matches.map((match) => (
              <button
                key={match.n_id}
                type="button"
                onClick={() => {
                  setResearcherQuery(match.name);
                  onSelectResearcher(match.n_id);
                }}
              >
                {match.name}
              </button>
            ))}
          </div>
        ) : null}
        {selectedResearcher !== null ? (
          <button
            type="button"
            className="atlas-ghost atlas-small"
            onClick={() => {
              setResearcherQuery("");
              onSelectResearcher(null);
            }}
          >
            Clear researcher
          </button>
        ) : null}
      </label>

      <label className="atlas-field">
        <span>Topic</span>
        <select
          value={selectedTopic ?? ""}
          onChange={(e) => {
            const value = e.target.value;
            onSelectTopic(value === "" ? null : Number(value));
          }}
        >
          <option value="">All topics</option>
          {topics.map((topic) => (
            <option key={topic.topic} value={topic.topic}>
              {topic.topic}. {topic.topic_name}
            </option>
          ))}
        </select>
      </label>

      <label className="atlas-field">
        <span>Policy Domain</span>
        <select
          value={selectedPolicy ?? ""}
          onChange={(e) => {
            const value = e.target.value;
            onSelectPolicy(value === "" ? null : Number(value));
          }}
        >
          <option value="">All policy domains</option>
          {policies.map((policy) => (
            <option key={policy.policy_domain} value={policy.policy_domain}>
              {policy.policy_name}
            </option>
          ))}
        </select>
      </label>

      <label className="atlas-field atlas-toggle">
        <span>Map layer</span>
        <div className="atlas-toggle-row">
          <label className="atlas-radio">
            <input
              type="radio"
              name="atlas-data-mode"
              value="publication"
              checked={dataMode === "publication"}
              onChange={() => onSelectDataMode("publication")}
            />
            Publications
          </label>
          <label className="atlas-radio">
            <input
              type="radio"
              name="atlas-data-mode"
              value="researcher"
              checked={dataMode === "researcher"}
              onChange={() => onSelectDataMode("researcher")}
            />
            Researchers
          </label>
        </div>
      </label>

      <label className="atlas-field atlas-toggle">
        <span>Color mode</span>
        <div className="atlas-toggle-row">
          <label className="atlas-radio">
            <input
              type="radio"
              name="atlas-color-mode"
              value="policy"
              checked={colorMode === "policy"}
              onChange={() => onSelectColorMode("policy")}
            />
            Policy
          </label>
          <label className="atlas-radio">
            <input
              type="radio"
              name="atlas-color-mode"
              value="topic"
              checked={colorMode === "topic"}
              onChange={() => onSelectColorMode("topic")}
            />
            Topic
          </label>
          <label className="atlas-radio">
            <input
              type="radio"
              name="atlas-color-mode"
              value="school"
              checked={colorMode === "school"}
              onChange={() => onSelectColorMode("school")}
            />
            School
          </label>
        </div>
      </label>

      <div className="atlas-field atlas-toggle">
        <span>Display</span>
        <div className="atlas-switch-list">
          <button
            type="button"
            className="atlas-switch-row"
            onClick={() => onToggleTopicNumbers(!showTopicNumbers)}
            aria-pressed={showTopicNumbers}
          >
            <span className="atlas-switch-label">Topic labels</span>
            <span className={`atlas-switch ${showTopicNumbers ? "on" : ""}`} aria-hidden="true" />
          </button>
          <button
            type="button"
            className="atlas-switch-row"
            onClick={() => onToggleHoverCard(!showHoverCard)}
            aria-pressed={showHoverCard}
          >
            <span className="atlas-switch-label">Hover cards</span>
            <span className={`atlas-switch ${showHoverCard ? "on" : ""}`} aria-hidden="true" />
          </button>
          <button
            type="button"
            className="atlas-switch-row"
            onClick={() => onToggleHideGreyedOutDots(!hideGreyedOutDots)}
            aria-pressed={hideGreyedOutDots}
          >
            <span className="atlas-switch-label">Hide greyed dots</span>
            <span className={`atlas-switch ${hideGreyedOutDots ? "on" : ""}`} aria-hidden="true" />
          </button>
        </div>
      </div>

      <div className="atlas-panel-footer">
        <p>
          Focus modes: All TRISS, single school, department, or researcher. Select a topic to isolate thematic clusters.
        </p>
      </div>
    </div>
  );
}
