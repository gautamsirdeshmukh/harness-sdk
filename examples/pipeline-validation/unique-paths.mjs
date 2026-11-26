// Deduplicate Linux workspace paths, preserving their order and exact spelling.
export function uniquePaths(paths) {
  const seen = new Set();
  return paths.filter(path => {
    const key = path.toLowerCase();
    if (seen.has(key)) return false;
    seen.add(key);
    return true;
  });
}
