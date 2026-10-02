import type { ReactElement, ReactNode } from 'react'
import type { DOMElement } from 'ink'

import type { ChatPanel, ChatPanelRow } from '../chat/controller.js'
import { PanelHelpFooter } from './help-footer.js'
import { Box, Text, useTheme } from './theme.js'

export interface PanelRowsProps {
  panel: ChatPanel
  rows: readonly ChatPanelRow[]
  selected: number
  start: number
  width: number
  pressedRow?: number
  hoveredRow?: number
  onPanelElement?: (element: DOMElement | null) => void
  onRowElement?: (index: number, element: DOMElement | null) => void
}

export function PanelItemHeader({
  label,
  active,
  pressed = false,
  clickable = false,
  bold = false,
  color,
  wrap = 'truncate-end',
}: {
  label: string
  active: boolean
  pressed?: boolean
  clickable?: boolean
  bold?: boolean
  color?: string
  wrap?: 'truncate-end' | 'wrap'
}): ReactElement {
  const { accent, hover } = useTheme()
  const markerColor = pressed && clickable ? hover : active ? (color ?? accent) : undefined
  const labelColor = pressed && clickable ? hover : active || pressed ? (color ?? accent) : undefined
  return (
    <Box flexShrink={1} overflow="hidden">
      <Box flexShrink={0}>
        <Text {...(markerColor ? { color: markerColor } : {})} dimColor={!active && !pressed}>
          {active && clickable ? '› ' : '  '}
        </Text>
      </Box>
      <Text {...(labelColor ? { color: labelColor } : {})} bold={bold || (active && !pressed)} wrap={wrap}>
        {label}
      </Text>
    </Box>
  )
}

export function PanelTitle({ title, color }: { title: string; color: string }): ReactElement {
  return (
    <Text color={color} bold wrap="truncate-end">
      {title}
    </Text>
  )
}

export function PanelContainer({
  width,
  height,
  inline = false,
  paddingX = 1,
  children,
  onElement,
}: {
  width: number
  height?: number
  inline?: boolean
  paddingX?: number
  children: ReactNode
  onElement?: (element: DOMElement | null) => void
}): ReactElement {
  const theme = useTheme()
  const frame = (
    <Box
      ref={onElement}
      width={width}
      {...(height === undefined ? {} : { height, overflow: 'hidden' })}
      flexDirection="column"
      paddingX={paddingX}
      {...(inline ? {} : { backgroundColor: theme.panel })}
    >
      {children}
      <PanelHelpFooter width={Math.max(1, width - paddingX * 2)} />
    </Box>
  )
  return inline ? (
    frame
  ) : (
    <Box position="absolute" width="100%" height="100%" alignItems="center" justifyContent="center" padding={1}>
      {frame}
    </Box>
  )
}
