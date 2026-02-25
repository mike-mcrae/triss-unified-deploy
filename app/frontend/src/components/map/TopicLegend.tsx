type LegendItem = {
  key: string;
  label: string;
  count: number;
};

type Props = {
  title: string;
  items: LegendItem[];
  colorMap: Record<string, number[]>;
  activeKey: string | null;
  onHoverItem: (key: string | null) => void;
  onClickItem: (key: string | null) => void;
  onClear: () => void;
};

export default function TopicLegend({ title, items, colorMap, activeKey, onHoverItem, onClickItem, onClear }: Props) {
  return (
    <div className="atlas-legend">
      <div className="atlas-legend-header">
        <h3>{title}</h3>
        {activeKey !== null ? (
          <button type="button" className="atlas-ghost atlas-small" onClick={onClear}>
            Clear
          </button>
        ) : null}
      </div>
      <div className="atlas-legend-list">
        {items.map((item) => {
          const color = colorMap[item.key] || [90, 90, 110];
          const isActive = activeKey === item.key;
          return (
            <button
              type="button"
              key={item.key}
              className={`atlas-legend-item ${isActive ? "active" : ""}`}
              onMouseEnter={() => onHoverItem(item.key)}
              onMouseLeave={() => onHoverItem(null)}
              onClick={() => onClickItem(item.key)}
            >
              <span
                className="atlas-swatch"
                style={{ background: `rgb(${color[0]}, ${color[1]}, ${color[2]})` }}
              />
              <span className="atlas-legend-text">{item.label}</span>
              <span className="atlas-legend-count">{item.count}</span>
            </button>
          );
        })}
      </div>
    </div>
  );
}
