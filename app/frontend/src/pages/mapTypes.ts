export type PointRecord = {
  key: string;
  n_id: number;
  publication_index?: number | null;
  name?: string;
  school?: string;
  department?: string;
  topic?: number | null;
  topic_name?: string | null;
  policy_domain?: number | null;
  policy_name?: string | null;
  x: number;
  y: number;
  title?: string;
  abstract?: string;
  email?: string;
  one_line_summary?: string;
  research_area?: string;
  topics?: string;
  subfields?: string;
  centroid_similarity?: number | null;
};

export type TopicsMap = Record<
  string,
  { topic_id: number; topic_name?: string; topic_description?: string; top_words?: string[] }
>;

export type FiltersData = {
  schools: string[];
  departments_by_school: Record<string, string[]>;
  researchers: { n_id: number; name: string; school?: string; department?: string }[];
  topics: { topic: number; topic_name?: string; count: number }[];
  policies: { policy_domain: number; policy_name?: string; count: number }[];
};

export type HoverInfo = {
  x: number;
  y: number;
  object: PointRecord | null;
};
