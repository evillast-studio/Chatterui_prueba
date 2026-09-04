import React, { useCallback, useEffect, useRef, useState } from 'react'
import { useTranslation } from 'react-i18next'
import { Animated, Easing, useAnimatedValue, View } from 'react-native'
import Markdown from 'react-native-markdown-display'

import ThemedButton from '@components/buttons/ThemedButton'
import AnimatedEllipsis from '@components/text/AnimatedEllipsis'
import { ChatSwipe } from '@db/schema'
import { useTextFilter } from '@lib/hooks/TextFilter'
import { useThrottledValue } from '@lib/hooks/useThrottledValue'
import { MarkdownStyle } from '@lib/markdown/Markdown'
import { Chats, useInference } from '@lib/state/Chat'

type ChatTextProps = {
    nowGenerating: boolean
    swipe: ChatSwipe
}

// While streaming, the buffer store updates on every generated token
// (often 15-30x/sec). Markdown parsing is not free - re-parsing the whole
// message on every single token wastes CPU/battery and can cause visible
// jank, all for updates the eye can't even perceive individually.
// Isolating the Markdown render behind React.memo means it only actually
// re-parses when the (throttled) text prop changes, not on every parent
// re-render.
type StreamingMarkdownProps = {
    text: string
    markdown: ReturnType<typeof MarkdownStyle.useCustomFormatting>['markdown']
    rules: ReturnType<typeof MarkdownStyle.useCustomFormatting>['rules']
    style: ReturnType<typeof MarkdownStyle.useCustomFormatting>['style']
}

const StreamingMarkdown = React.memo(
    ({ text, markdown, rules, style }: StreamingMarkdownProps) => (
        <Markdown mergeStyle={false} markdownit={markdown} rules={rules} style={style}>
            {text}
        </Markdown>
    ),
    (prev, next) =>
        prev.text === next.text &&
        prev.markdown === next.markdown &&
        prev.rules === next.rules &&
        prev.style === next.style
)

const ChatTextLast: React.FC<ChatTextProps> = ({ nowGenerating, swipe }) => {
    const { t } = useTranslation()
    const { markdown, rules, style } = MarkdownStyle.useCustomFormatting()

    const { buffer } = Chats.useBuffer()
    const [showHidden, setShowHidden] = useState(false)
    const viewRef = useRef<View>(null)
    const currentSwipeId = useInference((state) => state.currentSwipeId)
    const animHeight = useAnimatedValue(-1)
    const targetHeight = useRef(-1)
    const firstRender = useRef(true)

    const updateHeight = useCallback(() => {
        viewRef.current?.measure((_, __, ___, measuredHeight) => {
            if (firstRender.current) {
                firstRender.current = false
                animHeight.setValue(measuredHeight)
                return
            }
            const showPadding = nowGenerating && buffer.data
            const overflowPadding = showPadding ? 12 : 0
            const newHeight = measuredHeight + overflowPadding

            if (targetHeight.current === newHeight) return
            if (targetHeight.current > -1) animHeight.setValue(targetHeight.current)

            animHeight.stopAnimation(() =>
                Animated.timing(animHeight, {
                    toValue: newHeight,
                    duration: 300 * Math.max(1, Math.abs(newHeight - targetHeight.current) / 1000),
                    useNativeDriver: false,
                    easing: Easing.inOut((x) => x * x),
                }).start()
            )
            targetHeight.current = newHeight
        })
    }, [animHeight, buffer.data, nowGenerating])

    useEffect(() => {
        if (!nowGenerating && !firstRender.current) {
            setTimeout(() => updateHeight(), 400)
        }
    }, [nowGenerating, updateHeight])

    const filteredText = useTextFilter(swipe.swipe ?? '')
    const renderedText = showHidden ? swipe.swipe : filteredText.result

    const isStreamingThis = nowGenerating && swipe.id === currentSwipeId
    // ~12 updates/sec while streaming - smooth to the eye, far cheaper than
    // re-parsing markdown on every token. No throttling once generation
    // stops (or for messages that aren't the one currently streaming), so
    // the final text always lands exactly and immediately.
    const throttledBufferData = useThrottledValue(buffer.data, isStreamingThis ? 80 : 0)
    const displayText = isStreamingThis ? throttledBufferData.trim() : renderedText

    return (
        <Animated.View style={{ overflow: 'scroll', height: animHeight }}>
            <View style={{ minHeight: 10 }} ref={viewRef} onLayout={updateHeight}>
                {swipe.id === currentSwipeId && nowGenerating && buffer.data === '' && (
                    <AnimatedEllipsis />
                )}
                <StreamingMarkdown text={displayText} markdown={markdown} rules={rules} style={style} />
                {filteredText.found && (
                    <View style={{ flexDirection: 'row' }}>
                        <ThemedButton
                            onPress={() => setShowHidden(!showHidden)}
                            variant="secondary"
                            label={
                                showHidden
                                    ? t('chat.filteredText.hide')
                                    : t('chat.filteredText.show')
                            }
                            labelStyle={{ flex: 0, fontSize: 12 }}
                            buttonStyle={{
                                paddingVertical: 0,
                                paddingHorizontal: 0,
                                borderWidth: 0,
                            }}
                        />
                    </View>
                )}
            </View>
        </Animated.View>
    )
}

export default ChatTextLast
