import { fstatSync } from 'node:fs'

const SYNCHRONIZED_OUTPUT_START = '\u001b[?2026h'
const SYNCHRONIZED_OUTPUT_END = '\u001b[?2026l'
const SAVE_CURSOR = '\u001b7'
const RESTORE_CURSOR = '\u001b8'
const CURSOR_HOME = '\u001b[H'

function shareTerminal(output: NodeJS.WriteStream, errorOutput: NodeJS.WriteStream): boolean {
  if (output === errorOutput) return true
  if (!output.isTTY || !errorOutput.isTTY) return false
  const stdoutFd = 'fd' in output ? output.fd : undefined
  const stderrFd = 'fd' in errorOutput ? errorOutput.fd : undefined
  if (typeof stdoutFd !== 'number' || typeof stderrFd !== 'number') return false
  try {
    const stdout = fstatSync(stdoutFd)
    const stderr = fstatSync(stderrFd)
    return (
      stdout.isCharacterDevice() &&
      stderr.isCharacterDevice() &&
      stdout.dev === stderr.dev &&
      stdout.ino === stderr.ino &&
      stdout.rdev === stderr.rdev
    )
  } catch {
    return false
  }
}

export function createInkOutputs(
  output: NodeJS.WriteStream,
  errorOutput: NodeJS.WriteStream,
  alternateScreen: boolean
): { stdout: NodeJS.WriteStream; stderr: NodeJS.WriteStream } {
  const parkCursor = alternateScreen && output.isTTY
  // Apple Terminal leaves stale frames when it receives DEC synchronized-output markers.
  const stripSynchronization = process.env.TERM_PROGRAM === 'Apple_Terminal'
  if (!parkCursor && !stripSynchronization) return { stdout: output, stderr: errorOutput }

  let cursorSaved = false
  const wrap = (stream: NodeJS.WriteStream): NodeJS.WriteStream =>
    new Proxy(stream, {
      get(target, property): unknown {
        if (property === 'write') {
          return (chunk: unknown, ...args: unknown[]): unknown => {
            if (typeof chunk === 'string') {
              if (stripSynchronization) {
                chunk = chunk.replaceAll(SYNCHRONIZED_OUTPUT_START, '').replaceAll(SYNCHRONIZED_OUTPUT_END, '')
              }
              if (parkCursor && chunk && chunk !== SYNCHRONIZED_OUTPUT_START && chunk !== SYNCHRONIZED_OUTPUT_END) {
                // A bottom-row cursor makes the terminal scroll before delivering a height resize.
                // Both streams must resume from the last write to their shared terminal.
                chunk = `${cursorSaved ? RESTORE_CURSOR : ''}${chunk}${SAVE_CURSOR}${CURSOR_HOME}`
                cursorSaved = true
              }
            }
            return Reflect.apply(target.write, target, [chunk, ...args])
          }
        }
        const value = Reflect.get(target, property, target)
        return typeof value === 'function' ? value.bind(target) : value
      },
    })
  const stdout = wrap(output)
  const stderr =
    parkCursor && shareTerminal(output, errorOutput)
      ? errorOutput === output
        ? stdout
        : wrap(errorOutput)
      : errorOutput
  return { stdout, stderr }
}
