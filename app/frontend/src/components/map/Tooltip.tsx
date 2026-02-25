import type { PointRecord } from "../../pages/mapTypes";

type Props = {
  info: { x: number; y: number; object: PointRecord };
};

export default function Tooltip({ info }: Props) {
  const { x, y, object } = info;
  const abstract = object.abstract || "";
  const snippet = abstract.length > 420 ? `${abstract.slice(0, 420)}...` : abstract;
  const isResearcherPoint = object.publication_index === null || object.publication_index === undefined;

  return (
    <div className="atlas-tooltip" style={{ transform: `translate(${x + 16}px, ${y + 16}px)` }}>
      <div className="atlas-tooltip-title">{isResearcherPoint ? (object.name || "Researcher") : (object.title || "Untitled abstract")}</div>
      <div className="atlas-tooltip-meta">
        <span>{object.name || "Unknown researcher"}</span>
        <span>
          {object.school || ""}
          {object.department ? ` / ${object.department}` : ""}
        </span>
      </div>
      {isResearcherPoint && object.email ? (
        <div className="atlas-tooltip-meta">
          <span>{object.email}</span>
        </div>
      ) : null}
      <div className="atlas-tooltip-topic">
        {object.topic !== null && object.topic !== undefined ? (
          <span>
            Topic {object.topic}: {object.topic_name || "Unnamed"}
          </span>
        ) : (
          <span>No topic</span>
        )}
      </div>
      {isResearcherPoint ? (
        <>
          {object.research_area ? <p className="atlas-tooltip-abstract">{object.research_area}</p> : null}
          {object.one_line_summary ? <p className="atlas-tooltip-abstract">{object.one_line_summary}</p> : null}
          {object.topics ? <p className="atlas-tooltip-abstract"><strong>Themes:</strong> {object.topics}</p> : null}
        </>
      ) : (
        snippet ? <p className="atlas-tooltip-abstract">{snippet}</p> : null
      )}
    </div>
  );
}
