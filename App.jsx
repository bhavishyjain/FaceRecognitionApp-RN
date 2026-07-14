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
      if (onnxSession !== null) {
        setModelLoaded(true);
        return;
      }

      try {
        const modelAsset = require('./assets/models/arcface.onnx');
        const assetSource = Image.resolveAssetSource(modelAsset);
        const remoteUri = assetSource.uri;
        const cachedPath = `${RNFS.CachesDirectoryPath}/arcface.onnx`;

        console.log('[ONNX] URI:', remoteUri);
        console.log('[ONNX] Cache:', cachedPath);

        const isCached = await RNFS.exists(cachedPath);

        if (!isCached) {
          console.log('[ONNX] First launch — streaming to disk…');
          if (await RNFS.exists(cachedPath)) {
            await RNFS.unlink(cachedPath);
          }

          const task = RNFS.downloadFile({
            fromUrl: remoteUri,
            toFile: cachedPath,
            progressDivider: 1,
            begin: () => console.log('[ONNX] Download started'),
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
            throw new Error(`Download failed: HTTP ${result.statusCode}`);
          }
          console.log('[ONNX] Download complete — bytes:', result.bytesWritten);
        } else {
          console.log('[ONNX] Cache hit — loading from disk');
        }

        if (cancelled) {
          return;
        }
        console.log('[ONNX] Creating InferenceSession…');
        const session = await InferenceSession.create(`file://${cachedPath}`);
        if (cancelled) {
          return;
        }

        onnxSession = session;
        console.log(
          '[ONNX] Ready — inputs:',
          session.inputNames,
          'outputs:',
          session.outputNames,
        );
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
                  ? `Downloading model… ${loadProgress}%`
                  : 'Loading model from cache…'}
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
      {screen === 'verify' && onnxSession && (
        <VerifyScreen session={onnxSession} onBack={() => setScreen('home')} />
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
