import { useEffect, useRef, useState } from 'react'

/**
 * Returns a version of `value` that updates at most once every `intervalMs`.
 *
 * Purpose: streaming inference appends text to a Zustand store on *every*
 * token (potentially 15-30+ times/sec). If a component re-renders on every
 * one of those updates and feeds the result into something expensive
 * (e.g. a markdown parser that re-parses the *entire* message each time),
 * CPU/battery cost and UI jank scale with token rate for no visual benefit
 * - the eye can't perceive text updates faster than ~10-15fps anyway.
 *
 * Behavior:
 * - intervalMs <= 0: passthrough, no throttling (use when not streaming).
 * - Leading update: first change is applied immediately (no perceived delay).
 * - Trailing update: guarantees the final value is always applied, even if
 *   it arrives between throttle windows (no "stuck mid-token" text).
 */
export function useThrottledValue<T>(value: T, intervalMs: number): T {
    const [throttled, setThrottled] = useState(value)
    const lastRunRef = useRef(0)
    const timeoutRef = useRef<ReturnType<typeof setTimeout> | null>(null)
    const latestValueRef = useRef(value)

    latestValueRef.current = value

    useEffect(() => {
        if (intervalMs <= 0) {
            setThrottled(value)
            return
        }

        const now = Date.now()
        const elapsed = now - lastRunRef.current

        if (elapsed >= intervalMs) {
            lastRunRef.current = now
            setThrottled(value)
            return
        }

        if (timeoutRef.current) clearTimeout(timeoutRef.current)
        timeoutRef.current = setTimeout(() => {
            lastRunRef.current = Date.now()
            setThrottled(latestValueRef.current)
            timeoutRef.current = null
        }, intervalMs - elapsed)

        return () => {
            if (timeoutRef.current) clearTimeout(timeoutRef.current)
        }
    }, [value, intervalMs])

    // Always snap to the latest value the instant throttling is turned off
    // (e.g. generation just finished) so nothing is left stale.
    useEffect(() => {
        if (intervalMs <= 0) setThrottled(value)
    }, [intervalMs, value])

    return throttled
}
