export function formatDuration(seconds) {
  if (!Number.isFinite(seconds) || seconds < 0) throw new RangeError('Invalid duration');
  const whole = Math.floor(seconds);
  return `${Math.floor(whole / 60)}:${whole % 60}`;
}
