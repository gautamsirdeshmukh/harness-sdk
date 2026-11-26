import test from 'node:test';
import assert from 'node:assert/strict';
import { formatDuration } from './duration.mjs';
test('formats elapsed duration as m:ss', () => {
  for (const [seconds, expected] of [[0, '0:00'], [9, '0:09'], [59, '0:59'], [60, '1:00'], [61, '1:01'], [3601.9, '60:01']]) {
    assert.equal(formatDuration(seconds), expected);
  }
});
test('rejects invalid durations', () => {
  for (const seconds of [-1, NaN, Infinity]) assert.throws(() => formatDuration(seconds), RangeError);
});
