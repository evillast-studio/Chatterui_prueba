import { closeFd, getContentFd } from '@vali98/react-native-fs'
import {
    CompletionParams,
    ContextParams,
    initLlama,
    LlamaContext,
    RNLLAMA_MTMD_DEFAULT_MEDIA_MARKER,
} from 'llama.rn'
import { t } from 'i18next'
import { create } from 'zustand'
import { persist } from 'zustand/middleware'

import { ModelDataType } from '@db/schema'
import { Storage } from '@lib/enums/Storage'
import { AppDirectory, fileExists, readableFileSize, writeBase64File } from '@lib/utils/File'

import { checkGGMLDeprecated } from './GGML'
import { KV, Model } from './Model'
import { AppSettings } from '../../constants/GlobalValues'
import { Logger } from '../../state/Logger'
import { createMMKVStorage, mmkv } from '../../storage/MMKV'

export type CompletionTimings = {
    predicted_per_token_ms: number
    predicted_per_second: number | null
    predicted_ms: number
    predicted_n: number

    prompt_per_token_ms: number
    prompt_per_second: number | null
    prompt_ms: number
    prompt_n: number
}

export type CompletionOutput = {
    text: string
    timings: CompletionTimings
}

export type LlamaState = {
    context: LlamaContext | undefined
    model?: ModelDataType
    mmproj?: ModelDataType
    loadProgress: number
    chatCount: number
    promptCache?: string
    load: (model: ModelDataType) => Promise<void>
    loadMmproj: (model: ModelDataType) => Promise<void>
    setLoadProgress: (progress: number) => void
    unload: () => Promise<void>
    unloadMmproj: () => Promise<void>
    saveKV: (prompt: string | undefined, media_paths?: string[]) => Promise<void>
    loadKV: () => Promise<boolean>
    completion: (
        params: CompletionParams,
        callback: (text: string) => void,
        completed: (text: string, timngs: CompletionTimings) => void
    ) => Promise<void>
    stopCompletion: () => Promise<void>
    tokenLength: (text: string, mediaPaths?: string[]) => Promise<number>
    tokenize: (text: string, media_paths?: string[]) => Promise<{ tokens: number[] } | undefined>
}

export type LlamaConfig = {
    context_length: number
    threads: number
    gpu_layers: number
    batch: number
    ctx_shift: boolean
    devices: string[]
    // --- Advanced performance tuning (opt-in, toggleable in Settings) ---
    ubatch: number
    use_mmap: boolean
    use_mlock: boolean
    flash_attn_type: 'auto' | 'on' | 'off'
    cache_type_k: string
    cache_type_v: string
    kv_unified: boolean
    swa_full: boolean
    // --- Nuevos parámetros llama.rn 0.12.9 ---
    cpu_mask: string           // Ej: '4-7' para cores A73 en Kirin 710. Vacío = sin fijar
    no_extra_bufts: boolean    // Reduce RAM desactivando buffer de repack (algo más lento en prefill)
    rope_freq_base: number     // 0 = valor del modelo. >0 overrides (ej. 500000 para extender contexto)
}

export type EngineDataProps = {
    config: LlamaConfig
    lastModel?: ModelDataType
    lastMmproj?: ModelDataType
    setConfiguration: (config: LlamaConfig) => void
    setLastModelLoaded: (model: ModelDataType | undefined) => void
    setLastMmprojLoaded: (model: ModelDataType | undefined) => void
    maybeClearLastLoaded: (mode: ModelDataType) => void
}

const sessionFile = `${AppDirectory.SessionPath}llama-session.bin`

const defaultConfig = {
    context_length: 4096,
    threads: 4,
    gpu_layers: 0,
    batch: 512,
    ctx_shift: true,
    devices: [],
    // Defaults chosen to match this app's original (pre-toggle) behavior
    // exactly: use_mmap/use_mlock true, flash_attn off, cache in f16.
    // Nothing changes for anyone who doesn't touch these in Settings.
    ubatch: 512,
    use_mmap: true,
    use_mlock: true,
    flash_attn_type: 'off',
    cache_type_k: 'f16',
    cache_type_v: 'f16',
    kv_unified: true,
    swa_full: true,
    // Nuevos en llama.rn 0.12.9
    cpu_mask: '',              // Vacío = dejar que el SO decida
    no_extra_bufts: false,     // false = repack ON (más rápido en ARM)
    rope_freq_base: 0,         // 0 = usar el valor que trae el modelo
}

export namespace Llama {
    export const useLlamaPreferencesStore = create<EngineDataProps>()(
        persist(
            (set, get) => ({
                config: defaultConfig,
                setConfiguration: (config: LlamaConfig) => {
                    set({ config: config })
                },
                setLastModelLoaded: (model: ModelDataType | undefined) => {
                    if (get().lastModel?.id === model?.id) return
                    set({ lastModel: model, lastMmproj: undefined })
                },
                setLastMmprojLoaded: (mmproj: ModelDataType | undefined) => {
                    set({ lastMmproj: mmproj })
                },
                maybeClearLastLoaded: (data) => {
                    if (data.id === get().lastModel?.id) {
                        set({ lastModel: undefined, lastMmproj: undefined })
                    } else if (data.id === get().lastMmproj?.id) {
                        set({ lastMmproj: undefined })
                    }
                },
            }),
            {
                name: Storage.EngineData,
                partialize: (state) => ({
                    config: state.config,
                    lastModel: state.lastModel,
                    lastMmproj: state.lastMmproj,
                }),
                storage: createMMKVStorage(),
                version: 4,
                migrate: (persistedState: any, version) => {
                    if (version === 1) {
                        persistedState.config.ctx_shift = true
                        Logger.info('Migrated to v2 EngineData')
                    }
                    if (version === 2) {
                        persistedState.config.devices = []
                        Logger.info('Migrated to v3 EngineData')
                    }
                    if (version === 3) {
                        // Backfill advanced perf fields for existing users with
                        // values that reproduce prior (pre-toggle) behavior.
                        persistedState.config.ubatch = persistedState.config.batch ?? 512
                        persistedState.config.use_mmap = true
                        persistedState.config.use_mlock = true
                        // MIGRADO: flash_attn (bool) → flash_attn_type (string)
                        persistedState.config.flash_attn_type = persistedState.config.flash_attn ? 'on' : 'off'
                        delete persistedState.config.flash_attn
                        persistedState.config.cache_type_k = 'f16'
                        persistedState.config.cache_type_v = 'f16'
                        persistedState.config.kv_unified = true
                        persistedState.config.swa_full = true
                        // Nuevos campos llama.rn 0.12.9
                        persistedState.config.cpu_mask = ''
                        persistedState.config.no_extra_bufts = false
                        persistedState.config.rope_freq_base = 0
                        Logger.info('Migrated to v4 EngineData (llama.rn: flash_attn_type + new params)')
                    }
                    return persistedState
                },
            }
        )
    )

    export const useLlamaModelStore = create<LlamaState>()((set, get) => ({
        context: undefined,
        loadProgress: 0,
        chatCount: 0,
        promptCache: undefined,
        load: async (model: ModelDataType) => {
            const config = useLlamaPreferencesStore.getState().config

            if (get()?.model?.id === model.id) {
                return Logger.errorToast(t('model.toast.modelAlreadyLoaded'))
            }

            if (checkGGMLDeprecated(parseInt(model.quantization))) {
                return Logger.errorToast(t('model.toast.quantizationNoLongerSupported'))
            }

            if (!(await Model.getModelExists(model.file_path))) {
                Logger.errorToast(t('model.toast.modelDoesNotExist'))
                Model.verifyModelList()
                return
            }

            if (get().context !== undefined) {
                await get().unload()
            }

            let model_path = model.file_path
            if (model.file_path.includes('content://')) {
                model_path = (await getContentFd(model_path)) ?? model_path
            }

            const params: ContextParams = {
                model: model_path,
                n_ctx: config.context_length,
                n_threads: config.threads,
                n_batch: config.batch,
                n_ubatch: config.ubatch,
                ctx_shift: config.ctx_shift,
                n_gpu_layers: config.gpu_layers,
                use_mlock: config.use_mlock,
                use_mmap: config.use_mmap,
                devices: config.devices,
                flash_attn_type: config.flash_attn_type,
                cache_type_k: config.cache_type_k,
                cache_type_v: config.cache_type_v,
                kv_unified: config.kv_unified,
                swa_full: config.swa_full,
                // Nuevos en llama.rn 0.12.9
                ...(config.cpu_mask ? { cpu_mask: config.cpu_mask } : {}),
                no_extra_bufts: config.no_extra_bufts,
                ...(config.rope_freq_base > 0 ? { rope_freq_base: config.rope_freq_base } : {}),
            } as ContextParams

            Logger.info(
                `\n------ MODEL LOAD -----\n Model Name: ${model.name}\nStarting with parameters: \nContext Length: ${params.n_ctx}\nThreads: ${params.n_threads}\nBatch Size: ${params.n_batch}\nuBatch Size: ${config.ubatch}\nGPU Layers: ${params.n_gpu_layers}\nuse_mmap: ${config.use_mmap}\nuse_mlock: ${config.use_mlock}\nflash_attn_type: ${config.flash_attn_type}\ncache_type_k: ${config.cache_type_k}\ncache_type_v: ${config.cache_type_v}\nkv_unified: ${config.kv_unified}\nswa_full: ${config.swa_full}\ncpu_mask: ${config.cpu_mask || "(auto)"}\nno_extra_bufts: ${config.no_extra_bufts}\nrope_freq_base: ${config.rope_freq_base || "(modelo)"}\n`
            )

            const progressCallback = (progress: number) => {
                if (progress % 5 === 0) get().setLoadProgress(progress)
            }

            const llamaContext = await initLlama(params, progressCallback).catch((error) => {
                Logger.errorToast(t('model.toast.couldNotLoadModel'), JSON.stringify(error))
                if (model.file_path.includes('content://')) {
                    closeFd(model_path)
                }
            })

            if (!llamaContext) return

            set({
                context: llamaContext,
                model: model,
                chatCount: 1,
            })

            // updated EngineData
            useLlamaPreferencesStore.getState().setLastModelLoaded(model)
            KV.useKVStore.getState().setKvCacheLoaded(false)
        },
        loadMmproj: async (model: ModelDataType) => {
            const context = get().context
            if (!context) return

            let model_path = model.file_path
            if (model.file_path.includes('content://')) {
                model_path = (await getContentFd(model_path)) ?? model_path
            }

            Logger.info('Loading MMPROJ')
            await context.initMultimodal({ path: model_path, use_gpu: true }).catch((e) => {
                if (model.file_path.includes('content://')) {
                    closeFd(model_path)
                }

                Logger.errorToast(t('model.toast.failedToLoadMMPROJ'), JSON.stringify(e))
            })
            if (await context.isMultimodalEnabled()) {
                const capabilities = await context.getMultimodalSupport()
                Logger.info(
                    `MMPROJ Loaded:\n- Vision: ${capabilities.vision}\n- Audio: ${capabilities.audio}`
                )
            }

            set({
                mmproj: model,
            })

            useLlamaPreferencesStore.getState().setLastMmprojLoaded(model)
        },
        setLoadProgress: (progress: number) => {
            set({ loadProgress: progress })
        },
        unload: async () => {
            if (get().mmproj) {
                await get().context?.releaseMultimodal()
            }

            await get().context?.release()
            set({
                context: undefined,
                model: undefined,
                mmproj: undefined,
            })
            Logger.info('Model Unloaded')
        },
        unloadMmproj: async () => {
            if (!get().mmproj) return
            await get()
                .context?.releaseMultimodal()
                .catch((e) => {
                    Logger.errorToast(t('model.toast.failedToUnloadMMPROJ'), JSON.stringify(e))
                })
            set({
                mmproj: undefined,
            })
        },
        completion: async (
            params: CompletionParams,
            callback = (text: string) => {},
            completed = (text: string) => {}
        ) => {
            const llamaContext = get().context
            if (llamaContext === undefined) {
                Logger.errorToast(t('model.toast.noModelLoaded'))
                return
            }

            return llamaContext
                .completion(params, (data) => {
                    callback(data.token)
                })
                .then(async ({ text, timings }: CompletionOutput) => {
                    completed(text, timings)
                    Logger.info(
                        `\n---- Start Chat ${get().chatCount} ----\n${textTimings(timings)}\n---- End Chat ${get().chatCount} ----\n`
                    )
                    set({ chatCount: get().chatCount + 1 })
                    if (mmkv.getBoolean(AppSettings.SaveLocalKV)) {
                        await get().saveKV(params.prompt, params.media_paths ?? [])
                    }
                })
        },
        stopCompletion: async () => {
            await get().context?.stopCompletion()
        },
        saveKV: async (prompt, media_paths) => {
            const llamaContext = get().context
            if (!llamaContext) {
                Logger.errorToast(t('model.toast.noModelLoaded'))
                return
            }

            if (prompt) {
                const tokens = (await get().tokenize(prompt, media_paths ?? []))?.tokens
                KV.useKVStore.getState().setKvCacheTokens(tokens ?? [])
            }

            if (!fileExists(sessionFile)) {
                Logger.warn('Session file does not exist, creating...')
                await writeBase64File(sessionFile, '')
            }

            const now = performance.now()
            const data = await llamaContext.saveSession(sessionFile.replace('file://', ''))
            Logger.info(
                data === -1
                    ? 'Failed to save KV cache'
                    : `Saved KV in ${Math.floor(performance.now() - now)}ms with ${data} tokens`
            )
            Logger.info(`Current KV Size is: ${readableFileSize(await KV.getKVSize())}`)
        },
        loadKV: async () => {
            let result = false
            const llamaContext = get().context
            if (!llamaContext) {
                Logger.errorToast(t('model.toast.noModelLoaded'))
                return false
            }
            if (!fileExists(sessionFile)) {
                Logger.warn('No Cache found')
                return false
            }
            await llamaContext
                .loadSession(sessionFile.replace('file://', ''))
                .then(() => {
                    Logger.info('Session loaded from KV cache')
                    result = true
                })
                .catch(() => {
                    Logger.error('Session loaded could not load from KV cache')
                })
            return result
        },
        tokenLength: async (text: string, mediaPaths: string[] = []) => {
            const finalPaths = get().mmproj ? mediaPaths : []
            if (!get().mmproj && mediaPaths.length > 0) {
                Logger.warnToast(t('model.toast.mediaAddedWithoutMMPROJModel'))
            }
            const result = await get().context?.tokenize(
                text + finalPaths.map(() => RNLLAMA_MTMD_DEFAULT_MEDIA_MARKER).join(),
                {
                    media_paths: finalPaths.map((item) => item.replace('file://', '')),
                }
            )
            if (!result) return 0
            return result.tokens.length
        },
        tokenize: async (text: string, media_paths: string[] = []) => {
            const params = get().mmproj ? { media_paths } : {}
            return await get().context?.tokenize(text, params)
        },
    }))

    const textTimings = (timings: CompletionTimings) => {
        return (
            `\n[Prompt Timings]` +
            (timings.prompt_n > 0
                ? `\nPrompt Per Token: ${timings.prompt_per_token_ms.toFixed(2)} ms/token` +
                  `\nPrompt Per Second: ${timings.prompt_per_second?.toFixed(2) ?? 0} tokens/s` +
                  `\nPrompt Time: ${(timings.prompt_ms / 1000).toFixed(2)}s` +
                  `\nPrompt Tokens: ${timings.prompt_n} tokens`
                : '\nNo Tokens Processed') +
            `\n\n[Predicted Timings]` +
            (timings.predicted_n > 0
                ? `\nPredicted Per Token: ${timings.predicted_per_token_ms.toFixed(2)} ms/token` +
                  `\nPredicted Per Second: ${timings.predicted_per_second?.toFixed(2) ?? 0} tokens/s` +
                  `\nPrediction Time: ${(timings.predicted_ms / 1000).toFixed(2)}s` +
                  `\nPredicted Tokens: ${timings.predicted_n} tokens\n`
                : '\nNo Tokens Generated')
        )
    }
}
