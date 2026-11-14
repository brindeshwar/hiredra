// The screen registry: the sidebar and the router read only this list. Adding a feature is one new screen file and
// one line here.

import type { ComponentType } from 'react'
import type { Api } from '../api'
import { PipelineScreen } from './Pipeline'
import { JobsScreen } from './Jobs'
import { PacketsScreen } from './Packets'
import { BatchesScreen } from './Batches'
import { ApplyScreen } from './Apply'

export type ScreenProps = { api: Api; go: (id: string, arg?: string) => void; arg?: string }

export type Screen = { id: string; title: string; hint: string; component: ComponentType<ScreenProps> }

export const SCREENS: Screen[] = [
  { id: 'pipeline', title: 'Pipeline', hint: 'Run the stages, see counts and quota', component: PipelineScreen },
  { id: 'jobs', title: 'Jobs', hint: 'Ranked, review and the rest', component: JobsScreen },
  { id: 'packets', title: 'Packets', hint: 'Tailored resumes and cover letters', component: PacketsScreen },
  { id: 'batches', title: 'Batches', hint: 'Groups of 10 to review together', component: BatchesScreen },
  { id: 'apply', title: 'Apply', hint: 'Your turn: answers to copy, you submit', component: ApplyScreen }
]
