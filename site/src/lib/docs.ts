import { getCollection, type CollectionEntry } from 'astro:content';

export type DocEntry = CollectionEntry<'docs'>;

export interface SidebarItem {
  id: string;
  href: string;
  label: string;
  position: number;
}

// A category sub-group inside the Examples section (e.g. "Agents").
export interface SidebarCategory {
  label: string;
  position: number;
  items: SidebarItem[];
}

export interface SidebarGroup {
  label: string;
  position: number;
  /** Ungrouped items rendered directly under the group (no category). */
  items: SidebarItem[];
  /** Category sub-groups, each with their own items. */
  categories: SidebarCategory[];
}

// Human-friendly labels for the top-level folders in src/content/docs.
const GROUP_LABELS: Record<string, string> = {
  examples: 'Examples',
};

const GROUP_POSITIONS: Record<string, number> = {
  '': 0, // root-level docs (e.g. intro) come first
  examples: 2,
};

// Order categories appear within the Examples group. Categories not listed
// here fall to the end (position 100), sorted alphabetically as a tiebreak.
const CATEGORY_POSITIONS: Record<string, number> = {
  Agents: 10,
  'Developer Experience': 20,
  Apps: 30,
  Data: 40,
  Ingestion: 50,
};

const docHref = (id: string) => `/docs/${id === 'intro' ? '' : id + '/'}`;

const labelFor = (entry: DocEntry) =>
  entry.data.sidebar_label ?? entry.data.title;

/** All published docs, sorted for stable ordering. */
export async function getDocs(): Promise<DocEntry[]> {
  const docs = await getCollection(
    'docs',
    ({ data }: DocEntry) => import.meta.env.PROD !== true || !data.draft,
  );
  return docs.sort(
    (a, b) => a.data.sidebar_position - b.data.sidebar_position,
  );
}

/**
 * Build the sidebar tree: root-level docs become standalone items; docs inside
 * a folder are grouped under that folder's label; and within a folder, docs
 * that declare a `category` are further nested under that category. New
 * Markdown files appear automatically — folder = group, `category` = sub-group,
 * `sidebar_position` = order.
 */
export async function getSidebar(): Promise<SidebarGroup[]> {
  const docs = await getDocs();
  const groups = new Map<string, SidebarGroup>();

  for (const entry of docs) {
    const slug = entry.id;
    const folder = slug.includes('/') ? slug.split('/')[0] : '';

    if (!groups.has(folder)) {
      groups.set(folder, {
        label: GROUP_LABELS[folder] ?? '',
        position: GROUP_POSITIONS[folder] ?? 100,
        items: [],
        categories: [],
      });
    }
    const group = groups.get(folder)!;

    const item: SidebarItem = {
      id: slug,
      href: docHref(slug),
      label: labelFor(entry),
      position: entry.data.sidebar_position,
    };

    const category = entry.data.category;
    if (!category) {
      group.items.push(item);
      continue;
    }

    let cat = group.categories.find((c) => c.label === category);
    if (!cat) {
      cat = {
        label: category,
        position: CATEGORY_POSITIONS[category] ?? 100,
        items: [],
      };
      group.categories.push(cat);
    }
    cat.items.push(item);
  }

  const result = [...groups.values()];
  for (const g of result) {
    g.items.sort((a, b) => a.position - b.position);
    g.categories.sort(
      (a, b) => a.position - b.position || a.label.localeCompare(b.label),
    );
    g.categories.forEach((c) =>
      c.items.sort((a, b) => a.position - b.position),
    );
  }
  return result.sort((a, b) => a.position - b.position);
}
