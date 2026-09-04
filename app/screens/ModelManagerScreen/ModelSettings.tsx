import { useFocusEffect } from 'expo-router'
import React, { useEffect, useState } from 'react'
import { useTranslation } from 'react-i18next'
import { BackHandler, View } from 'react-native'
import { useMMKVBoolean, useMMKVNumber } from 'react-native-mmkv'
import Animated, { Easing, SlideInRight, SlideOutRight } from 'react-native-reanimated'
import { useShallow } from 'zustand/react/shallow'

import ThemedButton from '@components/buttons/ThemedButton'
import HorizontalSelector from '@components/input/HorizontalSelector'
import ThemedSlider from '@components/input/ThemedSlider'
import ThemedSwitch from '@components/input/ThemedSwitch'
import SectionTitle from '@components/text/SectionTitle'
import Alert from '@components/views/Alert'
import { AppSettings, Global } from '@lib/constants/GlobalValues'
import { Llama } from '@lib/engine/Local/LlamaLocal'
import { KV } from '@lib/engine/Local/Model'
import useBackendDevices from '@lib/hooks/BackendDevices'
import { Logger } from '@lib/state/Logger'
import { readableFileSize } from '@lib/utils/File'

type ModelSettingsProp = {
    modelImporting: boolean
    modelLoading: boolean
    exit: () => void
}

const deviceLabels = { GPUOpenCL: 'OpenCL', HTP0: 'Hexagon', CPU: 'CPU' }

const ModelSettings: React.FC<ModelSettingsProp> = ({ modelImporting, modelLoading, exit }) => {
    const { t } = useTranslation()
    const { config, setConfig } = Llama.useLlamaPreferencesStore(
        useShallow((state) => ({
            config: state.config,
            setConfig: state.setConfiguration,
        }))
    )

    const devices = useBackendDevices()

    const [saveKV, setSaveKV] = useMMKVBoolean(AppSettings.SaveLocalKV)
    const [autoloadLocal, setAutoloadLocal] = useMMKVBoolean(AppSettings.AutoLoadLocal)
    const [showModelInChat, setShowModelInChat] = useMMKVBoolean(AppSettings.ShowModelInChat)
    const [performanceMode, setPerformanceMode] = useMMKVBoolean(AppSettings.PerformanceMode)
    const [threadCount] = useMMKVNumber(Global.CPUThreads)

    const [kvSize, setKVSize] = useState(0)

    const getKVSize = async () => {
        const size = await KV.getKVSize()
        setKVSize(size)
    }

    useEffect(() => {
        KV.getKVSize().then(setKVSize)
    }, [])

    const backAction = () => {
        exit()
        return true
    }

    useFocusEffect(() => {
        const handler = BackHandler.addEventListener('hardwareBackPress', backAction)
        return () => handler.remove()
    })

    const handleDeleteKV = () => {
        Alert.alert({
            title: t('model.alert.deletekv.title'),
            description: t('model.alert.deletekv.description', { size: readableFileSize(kvSize) }),
            buttons: [
                { label: t('common.actions.delete') },
                {
                    label: t('model.alert.deletekv.title'),
                    onPress: async () => {
                        await KV.deleteKV()
                        Logger.info(t('model.toast.deletekv'))
                        getKVSize()
                    },
                    type: 'warning',
                },
            ],
        })
    }

    return (
        <Animated.ScrollView
            showsVerticalScrollIndicator={false}
            style={{ flex: 1 }}
            entering={SlideInRight.easing(Easing.inOut(Easing.cubic))}
            exiting={SlideOutRight.easing(Easing.inOut(Easing.cubic))}>
            <SectionTitle>{t('model.settings.cpu')}</SectionTitle>
            <View style={{ marginTop: 16 }} />
            {config && (
                <>
                    <ThemedSlider
                        label={t('model.maxcontext')}
                        value={config.context_length}
                        onValueChange={(value) => setConfig({ ...config, context_length: value })}
                        min={1024}
                        max={32768}
                        step={1024}
                        disabled={modelImporting || modelLoading}
                    />
                    <ThemedSlider
                        label={t('model.threads')}
                        value={config.threads}
                        onValueChange={(value) => setConfig({ ...config, threads: value })}
                        min={1}
                        max={threadCount ?? 8}
                        step={1}
                        disabled={modelImporting || modelLoading}
                    />

                    <ThemedSlider
                        label={t('model.batch')}
                        value={config.batch}
                        onValueChange={(value) => setConfig({ ...config, batch: value })}
                        min={16}
                        max={1024}
                        step={16}
                        disabled={modelImporting || modelLoading}
                    />

                    {/* GPU layers slider always shown - no compatibility gating.
                        On unsupported backends this can fail to load the model or
                        crash; if that happens, set it back to 0. */}
                    <ThemedSlider
                        label="GPU Layers (sin chequeo de compatibilidad)"
                        value={config.gpu_layers}
                        onValueChange={(value) => setConfig({ ...config, gpu_layers: value })}
                        min={0}
                        max={100}
                        step={1}
                        disabled={modelImporting || modelLoading}
                    />

                    <ThemedSwitch
                        label={t('model.contextshift')}
                        value={config.ctx_shift}
                        onChangeValue={(value) => {
                            setConfig({ ...config, ctx_shift: value })
                        }}
                    />

                    {devices.length > 1 && (
                        <HorizontalSelector
                            style={{ paddingBottom: 12 }}
                            label={t('model.backenddev')}
                            values={devices.map((item) => ({
                                label: deviceLabels[item as keyof typeof deviceLabels] ?? item,
                                value: item,
                            }))}
                            selected={config.devices?.[0]}
                            onPress={(value) => {
                                const devices = value === 'CPU' ? [value] : [value, 'CPU']
                                setConfig({ ...config, devices })
                            }}
                        />
                    )}
                </>
            )}
            <SectionTitle>Rendimiento avanzado</SectionTitle>
            <View style={{ marginTop: 16 }} />
            {config && (
                <>
                    <ThemedSlider
                        label="uBatch Size"
                        value={config.ubatch}
                        onValueChange={(value) => setConfig({ ...config, ubatch: value })}
                        min={16}
                        max={1024}
                        step={16}
                        disabled={modelImporting || modelLoading}
                    />
                    <ThemedSwitch
                        label="Usar mmap"
                        description="Mapea el modelo en memoria en vez de cargarlo completo. Carga más rápida, algo menos de control sobre RAM. Recomendado: activado."
                        value={config.use_mmap}
                        onChangeValue={(value) => setConfig({ ...config, use_mmap: value })}
                        disabled={modelImporting || modelLoading}
                    />
                    <ThemedSwitch
                        label="Usar mlock"
                        description="Bloquea el modelo en RAM para evitar que el sistema lo pagine a disco. Rendimiento más estable, pero usa más RAM de forma sostenida. Desactivalo en equipos con poca RAM."
                        value={config.use_mlock}
                        onChangeValue={(value) => setConfig({ ...config, use_mlock: value })}
                        disabled={modelImporting || modelLoading}
                    />
                    <ThemedSwitch
                        label="Flash Attention"
                        description="Puede acelerar la generación en hardware compatible. Si tu dispositivo/backend no lo soporta, puede fallar al cargar el modelo o no dar mejora — probalo y si falla, desactivalo."
                        value={config.flash_attn}
                        onChangeValue={(value) => setConfig({ ...config, flash_attn: value })}
                        disabled={modelImporting || modelLoading}
                    />
                    <HorizontalSelector
                        style={{ paddingBottom: 12 }}
                        label="Cache K (tipo)"
                        values={[
                            { label: 'f16 (más rápido)', value: 'f16' },
                            { label: 'q8_0 (menos RAM)', value: 'q8_0' },
                            { label: 'q4_0 (mín. RAM)', value: 'q4_0' },
                        ]}
                        selected={config.cache_type_k}
                        onPress={(value) => setConfig({ ...config, cache_type_k: value })}
                    />
                    <HorizontalSelector
                        style={{ paddingBottom: 12 }}
                        label="Cache V (tipo)"
                        values={[
                            { label: 'f16 (más rápido)', value: 'f16' },
                            { label: 'q8_0 (menos RAM)', value: 'q8_0' },
                            { label: 'q4_0 (mín. RAM)', value: 'q4_0' },
                        ]}
                        selected={config.cache_type_v}
                        onPress={(value) => setConfig({ ...config, cache_type_v: value })}
                    />
                    <ThemedSwitch
                        label="KV unificado (kv_unified)"
                        description="Buffer de atención unificado entre secuencias. Con una sola secuencia activa (uso normal de chat) no tiene downside; se puede desactivar solo si en algún momento usás decoding paralelo."
                        value={config.kv_unified}
                        onChangeValue={(value) => setConfig({ ...config, kv_unified: value })}
                        disabled={modelImporting || modelLoading}
                    />
                    <ThemedSwitch
                        label="SWA cache completo (swa_full)"
                        description="Usa caché de Sliding Window Attention a tamaño completo en vez de reducido. Evita artefactos de context-shift en modelos con SWA (ej. Gemma 3), a costa de algo más de RAM."
                        value={config.swa_full}
                        onChangeValue={(value) => setConfig({ ...config, swa_full: value })}
                        disabled={modelImporting || modelLoading}
                    />
                </>
            )}
            <SectionTitle>Modo rendimiento</SectionTitle>
            <ThemedSwitch
                label="Desactivar logging no esencial"
                description="Apaga los logs informativos (y su guardado en disco) que no son errores ni advertencias. Recomendado si solo te importa la velocidad; los mensajes de error/advertencia se siguen mostrando igual."
                value={performanceMode}
                onChangeValue={setPerformanceMode}
            />
            <SectionTitle>{t('model.settings.advanced')}</SectionTitle>
            <ThemedSwitch
                label={t('model.modelnamechat')}
                value={showModelInChat}
                onChangeValue={setShowModelInChat}
            />
            <ThemedSwitch
                label={t('model.autoload')}
                value={autoloadLocal}
                onChangeValue={setAutoloadLocal}
            />
            <ThemedSwitch
                label={t('model.savekv')}
                value={saveKV}
                onChangeValue={setSaveKV}
                description={saveKV ? '' : t('model.savekvdesc')}
            />
            {saveKV && (
                <ThemedButton
                    buttonStyle={{ marginTop: 8 }}
                    label={t('model.purgekv', { size: readableFileSize(kvSize) })}
                    onPress={handleDeleteKV}
                    variant={kvSize === 0 ? 'disabled' : 'critical'}
                />
            )}
        </Animated.ScrollView>
    )
}

export default ModelSettings
