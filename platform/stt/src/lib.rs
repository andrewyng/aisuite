//! A local, offline speech-to-text engine.
//!
//! This crate deliberately has no Tauri, UI, clipboard, or global-shortcut dependency. Hosts
//! own their own UX and permission flows; they use [`Dictation`] for microphone capture,
//! model provisioning, and final transcription.

use std::{
    fs,
    io::{self, Write},
    path::{Path, PathBuf},
    sync::{
        mpsc::{self, Receiver, Sender},
        Arc, Mutex,
    },
    thread,
};

use cpal::{
    traits::{DeviceTrait, HostTrait, StreamTrait},
    SampleFormat, Stream, StreamConfig,
};
use serde::Serialize;
use whisper_rs::{FullParams, SamplingStrategy, WhisperContext, WhisperContextParameters};

/// A reasonably fast English model for short OpenCoworker prompts (~142 MB).
pub const DEFAULT_MODEL_FILE: &str = "ggml-base.en.bin";
pub const DEFAULT_MODEL_URL: &str =
    "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-base.en.bin";
const WHISPER_SAMPLE_RATE: u32 = 16_000;

#[derive(Debug, Clone, Serialize)]
pub struct DictationStatus {
    pub recording: bool,
    pub model_installed: bool,
    pub model_name: &'static str,
}

struct Recording {
    stream: Stream,
    samples: Arc<Mutex<Vec<f32>>>,
    sample_rate: u32,
}

/// A reusable single-microphone dictation session manager.
///
/// It records only while a host has explicitly started a session; audio is held in memory for
/// that session and is never persisted. The downloaded recognition model is the only data kept
/// under `model_dir`.
pub struct Dictation {
    model_path: PathBuf,
    commands: Sender<Command>,
    recording: Arc<Mutex<bool>>,
}

enum Command {
    Start(Sender<Result<(), String>>),
    Stop(Sender<Result<RecordedAudio, String>>),
    Cancel(Sender<()>),
}

struct RecordedAudio {
    samples: Vec<f32>,
    sample_rate: u32,
}

impl Dictation {
    pub fn new(model_dir: impl Into<PathBuf>) -> Self {
        // CPAL's CoreAudio stream is intentionally !Send. Keep it on one dedicated owner thread
        // rather than unsafely forcing it through Tauri's Send + Sync application state.
        let (commands, receiver) = mpsc::channel();
        let recording = Arc::new(Mutex::new(false));
        let worker_recording = recording.clone();
        thread::spawn(move || capture_worker(receiver, worker_recording));
        Self {
            model_path: model_dir.into().join(DEFAULT_MODEL_FILE),
            commands,
            recording,
        }
    }

    pub fn status(&self) -> DictationStatus {
        DictationStatus {
            recording: self.recording.lock().map(|r| *r).unwrap_or(false),
            model_installed: self.model_path.is_file(),
            model_name: "Whisper Base English (local)",
        }
    }

    pub fn model_path(&self) -> &Path {
        &self.model_path
    }

    /// Downloads the default model atomically. Hosts should call this only after an explicit
    /// user action because it is a sizeable download.
    pub fn install_default_model(&self) -> Result<(), String> {
        if self.model_path.is_file() {
            return Ok(());
        }
        let parent = self
            .model_path
            .parent()
            .ok_or_else(|| "Could not determine the local model directory.".to_owned())?;
        fs::create_dir_all(parent).map_err(|e| format!("Could not create model directory: {e}"))?;

        let partial = self.model_path.with_extension("bin.part");
        let response = ureq::get(DEFAULT_MODEL_URL)
            .call()
            .map_err(|e| format!("Could not download the local voice model: {e}"))?;
        let mut input = response.into_reader();
        let mut output = fs::File::create(&partial)
            .map_err(|e| format!("Could not save the local voice model: {e}"))?;
        io::copy(&mut input, &mut output)
            .map_err(|e| format!("Could not save the local voice model: {e}"))?;
        output
            .flush()
            .map_err(|e| format!("Could not finish saving the local voice model: {e}"))?;
        fs::rename(&partial, &self.model_path)
            .map_err(|e| format!("Could not install the local voice model: {e}"))?;
        Ok(())
    }

    /// Begins microphone capture. A host must call [`stop_and_transcribe`](Self::stop_and_transcribe)
    /// or [`cancel`](Self::cancel) before a new recording can start.
    pub fn start(&self) -> Result<(), String> {
        let (reply, result) = mpsc::channel();
        self.commands
            .send(Command::Start(reply))
            .map_err(|_| "Dictation is unavailable because its audio worker stopped.".to_owned())?;
        result
            .recv()
            .map_err(|_| "Dictation is unavailable because its audio worker stopped.".to_owned())?
    }

    /// Stops capture and returns a final local transcript. This is intentionally synchronous so
    /// hosts can run it off their UI thread and decide how to present completion/error states.
    pub fn stop_and_transcribe(&self) -> Result<String, String> {
        let (reply, result) = mpsc::channel();
        self.commands
            .send(Command::Stop(reply))
            .map_err(|_| "Dictation is unavailable because its audio worker stopped.".to_owned())?;
        let RecordedAudio {
            samples,
            sample_rate,
        } = result
            .recv()
            .map_err(|_| "Dictation is unavailable because its audio worker stopped.".to_owned())??;
        if samples.len() < (sample_rate as usize / 4) {
            return Ok(String::new());
        }
        transcribe(&self.model_path, &resample_mono(&samples, sample_rate))
    }

    /// Discards the current in-memory recording without retaining or transcribing it.
    pub fn cancel(&self) {
        let (reply, done) = mpsc::channel();
        if self.commands.send(Command::Cancel(reply)).is_ok() {
            let _ = done.recv();
        }
    }
}

fn capture_worker(receiver: Receiver<Command>, recording_status: Arc<Mutex<bool>>) {
    let mut recording: Option<Recording> = None;
    for command in receiver {
        match command {
            Command::Start(reply) => {
                if recording.is_some() {
                    let _ = reply.send(Err("Dictation is already recording.".to_owned()));
                    continue;
                }
                match start_recording() {
                    Ok(next) => {
                        recording = Some(next);
                        if let Ok(mut active) = recording_status.lock() {
                            *active = true;
                        }
                        let _ = reply.send(Ok(()));
                    }
                    Err(error) => {
                        let _ = reply.send(Err(error));
                    }
                }
            }
            Command::Stop(reply) => {
                let result = recording
                    .take()
                    .ok_or_else(|| "Dictation is not recording.".to_owned())
                    .and_then(finish_recording);
                if let Ok(mut active) = recording_status.lock() {
                    *active = false;
                }
                let _ = reply.send(result);
            }
            Command::Cancel(reply) => {
                recording.take();
                if let Ok(mut active) = recording_status.lock() {
                    *active = false;
                }
                let _ = reply.send(());
            }
        }
    }
}

fn start_recording() -> Result<Recording, String> {
    let host = cpal::default_host();
    let device = host
        .default_input_device()
        .ok_or_else(|| "No microphone is available. Check your Mac sound settings.".to_owned())?;
    let supported = device
        .default_input_config()
        .map_err(|e| format!("Could not open the microphone: {e}"))?;
    let config: StreamConfig = supported.clone().into();
    let samples = Arc::new(Mutex::new(Vec::new()));
    let stream = build_stream(&device, &config, supported.sample_format(), samples.clone())?;
    stream
        .play()
        .map_err(|e| format!("Could not start microphone recording: {e}"))?;
    Ok(Recording {
        stream,
        samples,
        sample_rate: config.sample_rate.0,
    })
}

fn finish_recording(recording: Recording) -> Result<RecordedAudio, String> {
    let Recording {
        stream,
        samples,
        sample_rate,
    } = recording;
    drop(stream);
    let samples = samples
        .lock()
        .map_err(|_| "Could not read the recorded audio.".to_owned())?
        .clone();
    Ok(RecordedAudio {
        samples,
        sample_rate,
    })
}

fn build_stream(
    device: &cpal::Device,
    config: &StreamConfig,
    sample_format: SampleFormat,
    samples: Arc<Mutex<Vec<f32>>>,
) -> Result<Stream, String> {
    let channels = config.channels as usize;
    let on_error = |error| eprintln!("[ocw-stt] microphone stream error: {error}");
    match sample_format {
        SampleFormat::F32 => device
            .build_input_stream(
                config,
                move |data: &[f32], _| append_frames(&samples, data, channels, |sample| sample),
                on_error,
                None,
            )
            .map_err(|e| format!("Could not create microphone stream: {e}")),
        SampleFormat::I16 => device
            .build_input_stream(
                config,
                move |data: &[i16], _| {
                    append_frames(&samples, data, channels, |sample| sample as f32 / i16::MAX as f32)
                },
                on_error,
                None,
            )
            .map_err(|e| format!("Could not create microphone stream: {e}")),
        SampleFormat::U16 => device
            .build_input_stream(
                config,
                move |data: &[u16], _| {
                    append_frames(&samples, data, channels, |sample| {
                        (sample as f32 / u16::MAX as f32) * 2.0 - 1.0
                    })
                },
                on_error,
                None,
            )
            .map_err(|e| format!("Could not create microphone stream: {e}")),
        other => Err(format!("Unsupported microphone sample format: {other:?}")),
    }
}

fn append_frames<T>(target: &Arc<Mutex<Vec<f32>>>, data: &[T], channels: usize, convert: impl Fn(T) -> f32)
where
    T: Copy,
{
    let Ok(mut output) = target.lock() else {
        return;
    };
    output.reserve(data.len() / channels.max(1));
    for frame in data.chunks(channels.max(1)) {
        let sum: f32 = frame.iter().copied().map(&convert).sum();
        output.push(sum / frame.len() as f32);
    }
}

fn resample_mono(input: &[f32], source_rate: u32) -> Vec<f32> {
    if source_rate == WHISPER_SAMPLE_RATE {
        return input.to_vec();
    }
    let output_len = (input.len() as u64 * WHISPER_SAMPLE_RATE as u64 / source_rate as u64) as usize;
    let ratio = source_rate as f64 / WHISPER_SAMPLE_RATE as f64;
    (0..output_len)
        .map(|i| {
            let position = i as f64 * ratio;
            let left = position.floor() as usize;
            let right = (left + 1).min(input.len().saturating_sub(1));
            let fraction = (position - left as f64) as f32;
            input[left] * (1.0 - fraction) + input[right] * fraction
        })
        .collect()
}

fn transcribe(model_path: &Path, samples: &[f32]) -> Result<String, String> {
    if !model_path.is_file() {
        return Err("The local voice model is not installed yet.".to_owned());
    }
    let context = WhisperContext::new_with_params(
        model_path
            .to_str()
            .ok_or_else(|| "The local voice model path is not valid text.".to_owned())?,
        WhisperContextParameters::default(),
    )
    .map_err(|e| format!("Could not load the local voice model: {e}"))?;
    let mut state = context
        .create_state()
        .map_err(|e| format!("Could not prepare transcription: {e}"))?;
    let mut params = FullParams::new(SamplingStrategy::Greedy { best_of: 1 });
    params.set_language(Some("en"));
    params.set_translate(false);
    params.set_print_progress(false);
    params.set_print_special(false);
    params.set_print_realtime(false);
    params.set_suppress_blank(true);
    state
        .full(params, samples)
        .map_err(|e| format!("Could not transcribe the recording: {e}"))?;

    let mut text = String::new();
    for segment in state.as_iter() {
        let segment = segment
            .to_str()
            .map_err(|e| format!("Could not read the transcript: {e}"))?;
        text.push_str(segment);
    }
    Ok(text.trim().to_owned())
}

#[cfg(test)]
mod tests {
    use super::resample_mono;

    #[test]
    fn resampling_preserves_a_16khz_stream() {
        let input = vec![0.0, 0.5, -0.5];
        assert_eq!(resample_mono(&input, 16_000), input);
    }

    #[test]
    fn resampling_converts_duration() {
        let input = vec![0.0; 48_000];
        assert_eq!(resample_mono(&input, 48_000).len(), 16_000);
    }
}
