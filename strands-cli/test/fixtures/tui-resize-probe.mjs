import assert from 'node:assert/strict'
import { PassThrough } from 'node:stream'
import headless from '@xterm/headless'

// Install the same runtime hooks as the executable, before loading Ink.
await import('../../dist/src/tui/terminal/ink.js')
const { createElement } = await import('react')
const { render, renderToString } = await import('ink')
const { ChatController } = await import('../../src/tui/chat/controller.js')
const { runInkChat } = await import('../../src/tui/run.js')
const { ChatView } = await import('../../src/tui/view/chat-view.js')
const { sanitizeTerminalText } = await import('../../src/tui/terminal/sanitize.js')

const input = Object.assign(new PassThrough(), {
  isTTY: true,
  setRawMode() {},
  ref() {},
  unref() {},
})
const output = Object.assign(new PassThrough(), { isTTY: true, columns: 120, rows: 40 })
const screen = new headless.Terminal({
  cols: output.columns,
  rows: output.rows,
  allowProposedApi: true,
  convertEol: true,
})
let writes = []
output.on('data', (chunk) => {
  writes.push(chunk.toString())
  screen.write(chunk.toString())
})
let releaseStream
const streaming = new Promise((resolve) => {
  releaseStream = resolve
})
const controller = new ChatController(
  {
    id: 'resize',
    name: 'Resize fixture',
    protocol: 'strands',
    async *stream() {
      yield { type: 'textDelta', text: 'A response that wraps as the terminal width changes. '.repeat(12) }
      await streaming
      yield { type: 'textDelta', text: '\n\nThe response is complete.' }
      return { stopReason: 'endTurn' }
    },
    cancel() {},
  },
  { settings: { animations: false } }
)
let instance
const running = runInkChat(controller, {
  input,
  output,
  errorOutput: output,
  renderApp(element, options) {
    instance = render(element, { ...options, interactive: true, patchConsole: false })
    return instance
  },
})

async function flush() {
  await instance.waitUntilRenderFlush()
  await new Promise((resolve) => screen.write('', resolve))
}

function cells(terminal, rows, columns) {
  return Array.from({ length: rows }, (_, row) =>
    Array.from({ length: columns }, (_, column) => {
      const cell = terminal.buffer.active.getLine(row).getCell(column)
      return [
        cell.getChars(),
        cell.getFgColorMode(),
        cell.getFgColor(),
        cell.getBgColorMode(),
        cell.getBgColor(),
        cell.isBold(),
        cell.isDim(),
      ]
    })
  )
}

async function expectScreen(draft) {
  const frame = renderToString(
    createElement(ChatView, {
      snapshot: controller.getSnapshot(),
      input: draft,
      cursor: [...draft].length,
      terminalWidth: output.columns,
      terminalHeight: output.rows,
    }),
    { columns: output.columns }
  )
  assert.equal(screen.buffer.active.baseY, 0, 'resizing must not scroll the alternate screen')
  const reference = new headless.Terminal({
    cols: output.columns,
    rows: output.rows,
    allowProposedApi: true,
  })
  try {
    await new Promise((resolve) =>
      reference.write(
        frame
          .split('\n')
          .map((line, row) => `\x1b[${row + 1};1H${line}`)
          .join(''),
        resolve
      )
    )
    assert.deepEqual(
      cells(screen, output.rows, output.columns),
      cells(reference, output.rows, output.columns),
      'resized text, colors, and emphasis must match a fresh render'
    )
  } finally {
    reference.dispose()
  }
}

function resize(columns, rows) {
  const retainedRows = Math.min(rows, output.rows)
  const retainedColumns = Math.min(columns, output.columns)
  const retained = () => cells(screen, retainedRows, retainedColumns)
  const before = retained()
  screen.resize(columns, rows)
  assert.deepEqual(retained(), before, 'the terminal must not scroll the frame before delivering resize')
  output.columns = columns
  output.rows = rows
  output.emit('resize')
}

function expectResizePaint() {
  assert.equal(
    ['2J', '3J', '2K'].some((code) => writes.join('').includes(`\x1b[${code}`)),
    false,
    `resize to ${output.columns}x${output.rows} must not erase the screen or blank rows before repaint`
  )
  assert.equal(writes.filter((write) => sanitizeTerminalText(write).trim()).length, 1, 'paint only the settled layout')
  assert.equal(
    writes.some((write) => write.includes('\x1b[J') && !sanitizeTerminalText(write).trim()),
    false,
    'never paint a blank intermediate frame'
  )
}

try {
  await flush()
  let draft = 'keep this draft'
  input.write(draft)
  await flush()
  await expectScreen(draft)
  let turn
  for (const phase of ['welcome', 'streaming', 'completed']) {
    if (phase === 'streaming') {
      turn = controller.submit('A prompt with enough content to exercise transcript reflow.')
      await flush()
      assert.equal(controller.getSnapshot().status, 'running')
    }
    if (phase === 'completed') {
      releaseStream()
      await turn
      await flush()
    }
    for (const [columns, rows] of [
      [80, 24],
      [22, 10],
      [160, 50],
      [160, 25],
      [160, 50],
      [120, 40],
    ]) {
      writes = []
      resize(columns, rows)
      await flush()
      expectResizePaint()
      await expectScreen(draft)
    }
    writes = []
    resize(70, 25)
    resize(50, 18)
    resize(120, 40)
    await flush()
    await expectScreen(draft)
    assert.equal(writes.filter((write) => write.includes('\x1b[1;1H')).length, 1, 'coalesce resize bursts')

    writes = []
    resize(120, 40)
    await flush()
    assert.equal(writes.join(''), '', 'unchanged dimensions must not repaint')
    input.write('!')
    draft += '!'
    await flush()
    await expectScreen(draft)
  }
  controller.close(0)
  await running
  assert.equal(output.listenerCount('resize'), 0)
  process.stdout.write('resize frames, draft, transcript, burst coalescing, and cleanup verified\n')
} finally {
  releaseStream()
  controller.close(0)
  await running
  screen.dispose()
}
