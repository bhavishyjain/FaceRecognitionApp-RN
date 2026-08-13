import React, { useEffect, useState } from 'react';
import { Image, StatusBar, StyleSheet, Text, View } from 'react-native';
import { SafeAreaProvider, SafeAreaView } from 'react-native-safe-area-context';
import { Camera } from 'react-native-vision-camera';
import { InferenceSession } from 'onnxruntime-react-native';
import RNFS from 'react-native-fs';
import Svg, { Path, Rect } from 'react-native-svg';
import HomeScreen from './src/screens/HomeScreen';
import EnrollScreen from './src/screens/EnrollScreen';
import VerifyScreen from './src/screens/VerifyScreen';

let onnxSession = null;
let onnxFASSession = null;

async function copyBundledModelToCache(modelAsset, outputName, onProgress) {
  const assetSource = Image.resolveAssetSource(modelAsset);
  const sourceUri = assetSource.uri;
  const cachedPath = `${RNFS.CachesDirectoryPath}/${outputName}`;

  if (await RNFS.exists(cachedPath)) {
    return cachedPath;
  }

  const tmpPath = `${cachedPath}.tmp`;
  if (await RNFS.exists(tmpPath)) {
    await RNFS.unlink(tmpPath);
  }

  if (sourceUri.startsWith('http://') || sourceUri.startsWith('https://')) {
    const task = RNFS.downloadFile({
      fromUrl: sourceUri,
      toFile: tmpPath,
      progressDivider: 1,
      progress: res => {
        if (res.contentLength > 0 && onProgress) {
          onProgress(Math.round((res.bytesWritten / res.contentLength) * 100));
        }
      },
    });

    const result = await task.promise;
    if (result.statusCode < 200 || result.statusCode >= 300) {
      throw new Error(`${outputName} download failed: HTTP ${result.statusCode}`);
    }
  } else if (sourceUri.startsWith('file://')) {
    await RNFS.copyFile(decodeURI(sourceUri.replace('file://', '')), tmpPath);
  } else {
    await RNFS.copyFile(sourceUri, tmpPath);
  }

  await RNFS.moveFile(tmpPath, cachedPath);
  return cachedPath;
}

export default function App() {
  const [screen, setScreen] = useState('home');
  const [modelLoaded, setModelLoaded] = useState(false);
  const [modelError, setModelError] = useState(null);
  const [loadProgress, setLoadProgress] = useState(0);

  useEffect(() => {
    Camera.requestCameraPermission();
  }, []);

  useEffect(() => {
    let cancelled = false;

    async function loadModel() {
      if (onnxSession !== null && onnxFASSession !== null) {
        setModelLoaded(true);
        return;
      }

      try {
        const modelAsset = require('./assets/models/arcface.onnx');
        const assetSource = Image.resolveAssetSource(modelAsset);
        const remoteUri = assetSource.uri;
        const cachedPath = `${RNFS.CachesDirectoryPath}/arcface_nhwc_v2.onnx`;

        let isCached = await RNFS.exists(cachedPath);
        if (!isCached && !remoteUri.startsWith('http')) {
          await copyBundledModelToCache(modelAsset, 'arcface_nhwc_v2.onnx');
          isCached = true;
        }

        if (!isCached) {
          if (await RNFS.exists(cachedPath)) {
            await RNFS.unlink(cachedPath);
          }

          const task = RNFS.downloadFile({
            fromUrl: remoteUri,
            toFile: cachedPath,
            progressDivider: 1,
            progress: res => {
              if (cancelled) {
                return;
              }
              if (res.contentLength > 0) {
                setLoadProgress(
                  Math.round((res.bytesWritten / res.contentLength) * 100),
                );
              }
            },
          });

          const result = await task.promise;
          if (cancelled) {
            await RNFS.exists(cachedPath).then(e => {
              if (e) {
                RNFS.unlink(cachedPath);
              }
            });
            return;
          }
          if (result.statusCode !== 200) {
            throw new Error(
              `ArcFace download failed: HTTP ${result.statusCode}`,
            );
          }
        }

        if (cancelled) {
          return;
        }
        const fasAsset = require('./assets/models/minifasnet_v2.onnx');
        const fasSource = Image.resolveAssetSource(fasAsset);
        const fasRemoteUri = fasSource.uri;
        const fasCachedPath = `${RNFS.CachesDirectoryPath}/minifasnet_v2.onnx`;

        let isFasCached = await RNFS.exists(fasCachedPath);
        if (!isFasCached && !fasRemoteUri.startsWith('http')) {
          await copyBundledModelToCache(fasAsset, 'minifasnet_v2.onnx');
          isFasCached = true;
        }

        if (!isFasCached) {
          if (await RNFS.exists(fasCachedPath)) {
            await RNFS.unlink(fasCachedPath);
          }

          const task = RNFS.downloadFile({
            fromUrl: fasRemoteUri,
            toFile: fasCachedPath,
            progressDivider: 1,
            progress: () => {},
          });

          const result = await task.promise;
          if (cancelled) {
            await RNFS.exists(fasCachedPath).then(e => {
              if (e) {
                RNFS.unlink(fasCachedPath);
              }
            });
            return;
          }
          if (result.statusCode !== 200) {
            throw new Error(
              `MiniFASNet download failed: HTTP ${result.statusCode}`,
            );
          }
        }

        if (cancelled) {
          return;
        }

        const session = await InferenceSession.create(`file://${cachedPath}`);
        if (cancelled) {
          return;
        }
        onnxSession = session;

        const fasSession = await InferenceSession.create(
          `file://${fasCachedPath}`,
        );
        if (cancelled) {
          return;
        }
        onnxFASSession = fasSession;

        setModelLoaded(true);
      } catch (e) {
        console.error('[ONNX] Failed:', e);
        if (!cancelled) {
          setModelError(String(e));
        }
      }
    }

    loadModel();
    return () => {
      cancelled = true;
    };
  }, []);

  if (!modelLoaded) {
    return (
      <SafeAreaProvider>
        <StatusBar barStyle="light-content" backgroundColor={C.surface} />
        <View style={s.loadingRoot}>
          <View style={s.loadingIconRing}>
            <Svg
              width={36}
              height={36}
              viewBox="0 0 24 24"
              fill="none"
              stroke={C.accent}
              strokeWidth={1.5}
              strokeLinecap="round"
              strokeLinejoin="round"
            >
              <Rect x="4" y="4" width="16" height="16" rx="2" />
              <Rect x="9" y="9" width="6" height="6" rx="1" />
              <Path d="M15 2v2" />
              <Path d="M15 20v2" />
              <Path d="M2 15h2" />
              <Path d="M2 9h2" />
              <Path d="M20 15h2" />
              <Path d="M20 9h2" />
              <Path d="M9 2v2" />
              <Path d="M9 20v2" />
            </Svg>
          </View>

          <Text style={s.loadingTitle}>Preparing Face Recognition</Text>

          {modelError ? (
            <Text style={s.loadingError}>{modelError}</Text>
          ) : (
            <>
              <Text style={s.loadingSubtitle}>
                {loadProgress > 0
                  ? `Downloading models… ${loadProgress}%`
                  : 'Loading models from cache…'}
              </Text>

              <View style={s.progressTrack}>
                <View
                  style={[
                    s.progressFill,
                    {
                      width: `${Math.max(
                        loadProgress > 0 ? loadProgress : 8,
                        8,
                      )}%`,
                    },
                  ]}
                />
              </View>

              <Text style={s.loadingHint}>
                {loadProgress > 0
                  ? 'Only needed on first launch — saved for future use'
                  : 'This will only take a moment…'}
              </Text>
            </>
          )}
        </View>
      </SafeAreaProvider>
    );
  }

  return (
    <SafeAreaProvider>
      <StatusBar barStyle="light-content" backgroundColor={C.surface} />
      {screen === 'home' && (
        <SafeAreaView style={s.root} edges={['top', 'bottom']}>
          <HomeScreen
            onEnroll={() => setScreen('enroll')}
            onVerify={() => setScreen('verify')}
          />
        </SafeAreaView>
      )}
      {screen === 'enroll' && onnxSession && (
        <EnrollScreen session={onnxSession} onDone={() => setScreen('home')} />
      )}
      {screen === 'verify' && onnxSession && onnxFASSession && (
        <VerifyScreen
          session={onnxSession}
          fasSession={onnxFASSession}
          onBack={() => setScreen('home')}
        />
      )}
    </SafeAreaProvider>
  );
}

const C = {
  surface: '#09090B',
  card: '#18181B',
  cardElevated: '#27272A',
  border: '#27272A',
  primary: '#FAFAFA',
  primaryMuted: '#A1A1AA',
  accent: '#3B82F6',
  success: '#22C55E',
  danger: '#EF4444',
  text: '#FAFAFA',
  textSecondary: '#A1A1AA',
  textTertiary: '#52525B',
};

const s = StyleSheet.create({
  root: { flex: 1, backgroundColor: C.surface },

  loadingRoot: {
    flex: 1,
    backgroundColor: C.surface,
    alignItems: 'center',
    justifyContent: 'center',
    paddingHorizontal: 36,
  },
  loadingIconRing: {
    width: 90,
    height: 90,
    borderRadius: 45,
    backgroundColor: C.accent + '15',
    borderWidth: 2,
    borderColor: C.accent,
    alignItems: 'center',
    justifyContent: 'center',
    marginBottom: 28,
  },
  loadingTitle: {
    color: C.text,
    fontSize: 22,
    lineHeight: 30,
    fontWeight: '800',
    textAlign: 'center',
    marginBottom: 10,
  },
  loadingSubtitle: {
    color: C.textSecondary,
    fontSize: 14,
    lineHeight: 20,
    textAlign: 'center',
    marginBottom: 22,
  },
  loadingHint: {
    color: C.textTertiary,
    fontSize: 11,
    lineHeight: 16,
    textAlign: 'center',
  },
  loadingError: {
    color: C.danger,
    fontSize: 13,
    lineHeight: 18,
    textAlign: 'center',
    marginTop: 12,
  },

  progressTrack: {
    width: '100%',
    height: 6,
    backgroundColor: C.card,
    borderRadius: 3,
    overflow: 'hidden',
    marginBottom: 14,
  },
  progressFill: {
    height: '100%',
    backgroundColor: C.accent,
    borderRadius: 3,
  },
});
