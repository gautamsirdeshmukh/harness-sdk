import test from 'node:test';
import assert from 'node:assert/strict';
import { uniquePaths } from './unique-paths.mjs';
test('removes repeated paths', () => {
  assert.deepEqual(uniquePaths(['/src/a.ts', '/src/b.ts', '/src/a.ts']), ['/src/a.ts', '/src/b.ts']);
});
