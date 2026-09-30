import { afterEach, expect, it, vi } from 'vitest'

import { main } from '../src/cli/run.js'
import { SETUP_VERSION, CliConfigStore } from '../src/tui/config.js'

const runInkChat = vi.hoisted(() => vi.fn(async () => 0))
vi.mock('../src/tui/terminal/ink.js', () => ({}))
vi.mock('../src/tui/run.js', () => ({ runInkChat }))

afterEach(() => {
  vi.restoreAllMocks()
  runInkChat.mockClear()
  process.exitCode = undefined
})

it.each([
  { name: 'configured default', settings: {}, onboardingVersion: SETUP_VERSION, args: [], expected: false },
  {
    name: 'saved setup preference',
    settings: { setupOnLaunch: true },
    onboardingVersion: SETUP_VERSION,
    args: [],
    expected: false,
  },
  {
    name: 'first launch',
    settings: {},
    onboardingVersion: 0,
    args: [],
    expected: false,
  },
  {
    name: 'explicit setup',
    settings: {},
    onboardingVersion: SETUP_VERSION,
    args: ['--setup'],
    expected: true,
  },
  {
    name: 'explicit agent',
    settings: {},
    onboardingVersion: SETUP_VERSION,
    args: ['--agent', './agent.ts'],
    expected: false,
  },
  {
    name: 'explicit setup and agent',
    settings: {},
    onboardingVersion: SETUP_VERSION,
    args: ['--setup', '--agent', './agent.ts'],
    expected: true,
  },
])('selects startup setup for $name', async ({ settings, onboardingVersion, args, expected }) => {
  vi.spyOn(CliConfigStore, 'load').mockResolvedValue(CliConfigStore.memory({}, settings, { onboardingVersion }))
  const stdinIsTTY = process.stdin.isTTY
  const stdoutIsTTY = process.stdout.isTTY
  process.stdin.isTTY = true
  process.stdout.isTTY = true
  try {
    await main(args)
    expect(runInkChat).toHaveBeenCalledWith(expect.any(Function), expect.objectContaining({ setup: expected }))
    expect(process.exitCode).toBe(0)
  } finally {
    process.stdin.isTTY = stdinIsTTY
    process.stdout.isTTY = stdoutIsTTY
  }
})
