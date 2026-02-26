"""
spectrogram_sonification.py
----------------------------
Audio-visual sonification module for solar radio burst spectrograms from the
e-CALLISTO network (http://www.e-callisto.org/).

Maps spectrogram intensity and radio frequency to musical tones using additive
wind-chime synthesis, producing synchronised audio and animated MP4 video output.

Provides
--------
  - SpectrogramSonifier   : configurable additive synthesis engine
  - play_spectrogram_with_audio() : animated spectrogram with synced audio / video
  - MP3 mixing utilities  : intensity-gated background music blending

Intended to be used alongside callisto_pipeline.py for full event processing.

Repository
----------
  https://github.com/studentofstars/PyCallisto-SRB-Sonification

Dependencies
------------
  pyCallisto, numpy, matplotlib, scipy
  Optional: pydub (MP3 support), ffmpeg (MP4 export)

  Install core deps:  pip install -r requirements.txt
  Install ffmpeg:     sudo apt install ffmpeg   (Linux)
                      brew install ffmpeg       (macOS)

References
----------
  Pawase, R. & Raja, K. S. (2020). pyCallisto: A Python Library To Process
  The CALLISTO Spectrometer Data. arXiv:2006.16300 [astro-ph.IM]
  https://arxiv.org/abs/2006.16300

  pyCallisto on GitHub : https://github.com/ravipawase/pyCallisto
  pyCallisto on PyPI   : https://pypi.org/project/pyCallisto/
  e-CALLISTO network   : http://www.e-callisto.org/

Contributors
------------
  Mr. Ravindra Pawase
      Data Scientist, Cummins Inc., Pune 411 004, India
      ravi.pawase@gmail.com

  Dr. K. Sasikumar Raja
      Assistant Professor, Indian Institute of Astrophysics,
      II Block, Koramangala, Bengaluru 560 034, India
      sasikumarraja@gmail.com

  Mr. Mrutyunjaya Muduli
      B.E. Computer Science and Engineering,
      HKBK College of Engineering, Nagavara, Bengaluru 560 045, India
      mudulimrutyunjaya42@gmail.com

License
-------
  MIT — see LICENSE
"""

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib import cm
from matplotlib.animation import FuncAnimation
from scipy.io.wavfile import write as wav_write
import numpy as np
import subprocess
import datetime as dt
import tempfile
import shutil
import os

# Try to import pydub for MP3 handling
try:
    from pydub import AudioSegment
    PYDUB_AVAILABLE = True
except ImportError:
    PYDUB_AVAILABLE = False
    print("⚠️ pydub not installed. Run: pip install pydub")

# ================== DEFAULT CONFIGURATION ==================
MAPPING_MODE = 'pentatonic'  # Options: 'pentatonic', 'major', 'minor', 'blues', 'japanese', 'auld_lang_syne', 'proportional'
AUDIO_DURATION = 15  # Duration of audio in seconds
INTENSITY_THRESHOLD = 0.35  # Only sonify pixels above this threshold (0.0 to 1.0)
MIN_AUDIBLE_FREQ = 800  # Minimum audible frequency (Hz) - higher for softer sound
MAX_AUDIBLE_FREQ = 4000  # Maximum audible frequency (Hz) - higher pitched
SAVE_VIDEO = True  # Save as MP4 file
VIDEO_FILENAME = "solar_radio_chimes.mp4"

# Custom MP3 settings
USE_CUSTOM_MP3 = False  # Set to True to use a custom MP3 background
CUSTOM_MP3_PATH = ""  # Path to your MP3 file
MP3_VOLUME = 0.8  # Max volume of MP3 during high intensity (0.0 to 1.0)
SONIFICATION_VOLUME = 0.2  # Volume of sonification (0.0 to 1.0)


# ================== MP3 MIXING FUNCTIONS ==================
def load_mp3_as_numpy(mp3_path, target_duration, sample_rate=44100):
    """Load MP3 file and convert to numpy array"""
    if not PYDUB_AVAILABLE:
        print("⚠️ pydub not available, skipping MP3 loading")
        return None
    
    if not os.path.exists(mp3_path):
        print(f"⚠️ MP3 file not found: {mp3_path}")
        return None
    
    try:
        print(f"🎵 Loading MP3: {mp3_path}")
        audio = AudioSegment.from_mp3(mp3_path)
        
        # Convert to mono and set sample rate
        audio = audio.set_channels(1)
        audio = audio.set_frame_rate(sample_rate)
        
        # Get numpy array
        samples = np.array(audio.get_array_of_samples(), dtype=np.float32)
        samples = samples / 32768.0  # Normalize to -1.0 to 1.0
        
        # Adjust length to match target duration
        target_samples = int(target_duration * sample_rate)
        
        if len(samples) < target_samples:
            # Loop the audio if too short
            repeats = (target_samples // len(samples)) + 1
            samples = np.tile(samples, repeats)
        
        # Trim to exact length
        samples = samples[:target_samples]
        
        print(f"   ✓ Loaded {len(samples)/sample_rate:.1f}s of audio")
        return samples
        
    except Exception as e:
        print(f"⚠️ Error loading MP3: {e}")
        return None


def mix_audio(sonification, mp3_audio, sonification_vol=0.6, mp3_vol=0.4):
    """Mix sonification with MP3 background"""
    if mp3_audio is None:
        return sonification
    
    # Ensure same length
    min_len = min(len(sonification), len(mp3_audio))
    sonification = sonification[:min_len]
    mp3_audio = mp3_audio[:min_len]
    
    # Mix with volume levels
    mixed = (sonification * sonification_vol) + (mp3_audio * mp3_vol)
    
    # Normalize
    if np.max(np.abs(mixed)) > 0:
        mixed = mixed / np.max(np.abs(mixed)) * 0.95
    
    print(f"   ✓ Mixed audio: sonification={sonification_vol:.0%}, MP3={mp3_vol:.0%}")
    return mixed


def mix_audio_intensity_gated(sonification, mp3_audio, spectrogram_data, 
                               sonification_vol=0.2, mp3_vol=0.8, 
                               intensity_threshold=0.35, sample_rate=44100):
    """
    MP3 only plays during high intensity regions.
    The MP3 volume is modulated by the spectrogram intensity.
    Chimes/sonification is disabled.
    """
    if mp3_audio is None:
        print("   ⚠️ No MP3 audio available")
        return np.zeros(len(sonification))
    
    # Use MP3 length as reference
    min_len = len(mp3_audio)
    
    # Normalize spectrogram data
    spec_min = np.nanmin(spectrogram_data)
    spec_max = np.nanmax(spectrogram_data)
    normalized = (spectrogram_data - spec_min) / (spec_max - spec_min + 1e-10)
    normalized = np.nan_to_num(normalized, nan=0.0)
    
    # Get intensity envelope (max intensity across all frequencies at each time step)
    intensity_envelope = np.max(normalized, axis=0)
    
    # Apply threshold - only play MP3 when intensity exceeds threshold
    intensity_envelope = np.where(intensity_envelope >= intensity_threshold, 
                                   intensity_envelope, 0)
    
    # Stretch the envelope to match audio length
    n_time_steps = len(intensity_envelope)
    samples_per_step = min_len // n_time_steps
    
    # Create smooth gain envelope for MP3
    mp3_gain = np.zeros(min_len)
    for t_idx in range(n_time_steps):
        t_start = t_idx * samples_per_step
        t_end = min((t_idx + 1) * samples_per_step, min_len)
        mp3_gain[t_start:t_end] = intensity_envelope[t_idx]
    
    # Fill any remaining samples
    if n_time_steps * samples_per_step < min_len:
        mp3_gain[n_time_steps * samples_per_step:] = intensity_envelope[-1]
    
    # Smooth the gain envelope to avoid clicks (apply simple moving average)
    window_size = int(0.05 * sample_rate)  # 50ms smoothing window
    if window_size > 1:
        kernel = np.ones(window_size) / window_size
        mp3_gain = np.convolve(mp3_gain, kernel, mode='same')
    
    # Scale gain to max volume
    if np.max(mp3_gain) > 0:
        mp3_gain = mp3_gain / np.max(mp3_gain) * mp3_vol
    
    # Only MP3 gated by intensity (no chimes)
    mixed = mp3_audio * mp3_gain
    
    # Normalize
    if np.max(np.abs(mixed)) > 0:
        mixed = mixed / np.max(np.abs(mixed)) * 0.95
    
    # Calculate how much MP3 played
    active_time = np.sum(mp3_gain > 0.01) / sample_rate
    total_time = min_len / sample_rate
    print(f"   ✓ Intensity-gated mix: chimes={sonification_vol:.0%}, MP3 (gated)={mp3_vol:.0%}")
    print(f"   ✓ MP3 active for {active_time:.1f}s / {total_time:.1f}s ({100*active_time/total_time:.1f}% of duration)")
    
    return mixed


# ================== SONIFICATION CLASS ==================
class SpectrogramSonifier:
    """Converts spectrogram data to audio using additive synthesis"""
    
    # Musical scale definitions (ratios relative to base frequency)
    SCALES = {
        # Pentatonic scales - 5 notes, no dissonance
        'pentatonic': [1, 9/8, 5/4, 3/2, 5/3, 2],  # Major pentatonic
        'pentatonic_minor': [1, 6/5, 4/3, 3/2, 9/5, 2],  # Minor pentatonic
        
        # Diatonic scales - 7 notes
        'major': [1, 9/8, 5/4, 4/3, 3/2, 5/3, 15/8, 2],  # Major (Ionian)
        'minor': [1, 9/8, 6/5, 4/3, 3/2, 8/5, 9/5, 2],  # Natural minor (Aeolian)
        
        # Blues & Jazz
        'blues': [1, 6/5, 4/3, 45/32, 3/2, 9/5, 2],  # Blues scale with blue note
        
        # World music scales
        'japanese': [1, 16/15, 4/3, 3/2, 8/5, 2],  # Japanese Hirajoshi
        
        # Famous song-inspired tunings
        'auld_lang_syne': [1, 9/8, 5/4, 3/2, 5/3, 2],  # Perfect for New Year! 
    }
    
    def __init__(self, mapping_mode='pentatonic'):
        self.mapping_mode = mapping_mode
    
    def generate_audio(self, spectrogram_data, frequencies, duration=10, sample_rate=44100, intensity_threshold=0.0):
        """Generate audio from spectrogram data"""
        n_freq_bins, n_time_steps = spectrogram_data.shape
        total_samples = int(duration * sample_rate)
        samples_per_step = total_samples // n_time_steps
        
        audio = np.zeros(total_samples)
        
        # Normalize spectrogram data
        spec_min = np.nanmin(spectrogram_data)
        spec_max = np.nanmax(spectrogram_data)
        normalized = (spectrogram_data - spec_min) / (spec_max - spec_min + 1e-10)
        normalized = np.nan_to_num(normalized, nan=0.0)
        
        # Apply intensity threshold - set values below threshold to 0
        normalized_thresholded = np.where(normalized >= intensity_threshold, normalized, 0)
        
        # Map radio frequencies to audible range based on mapping mode
        if self.mapping_mode == 'pentatonic':
            audible_freqs = self._map_to_pentatonic(n_freq_bins)
        else:
            audible_freqs = self._map_proportional(frequencies)
        
        # Create wind chime envelope (gentle attack, long sustain, slow decay)
        t_segment = np.arange(samples_per_step) / sample_rate
        duration_sec = samples_per_step / sample_rate
        
        # Chime envelope: softer attack, longer ring
        decay_rate = 3.0  # Slower decay for wind chimes (they ring longer)
        chime_envelope = np.exp(-decay_rate * t_segment / duration_sec)
        # Gentle attack
        attack_samples = int(0.01 * sample_rate)  # 10ms gentle attack
        if attack_samples > 0 and attack_samples < len(chime_envelope):
            chime_envelope[:attack_samples] *= np.linspace(0, 1, attack_samples) ** 0.5
        
        print(f"   Processing {n_time_steps} time steps...")
        print(f"   🎐 Using wind chime synthesis - gentle & ethereal!")
        active_count = 0
        
        for t_idx in range(n_time_steps):
            if t_idx % 100 == 0:
                print(f"   Progress: {t_idx}/{n_time_steps} ({100*t_idx/n_time_steps:.1f}%)")
            
            column = normalized_thresholded[:, t_idx]
            t_start = t_idx * samples_per_step
            t_end = min((t_idx + 1) * samples_per_step, total_samples)
            
            chunk = np.zeros(t_end - t_start)
            chunk_len = t_end - t_start
            t_chunk = t_segment[:chunk_len]
            env_chunk = chime_envelope[:chunk_len]
            
            active_mask = column > 0.05
            active_indices = np.where(active_mask)[0]
            
            if len(active_indices) > 0:
                active_count += 1
            
            for i in active_indices:
                amplitude = column[i]
                freq = audible_freqs[i]
                scaled_amp = amplitude ** 0.4  # Even less compression for delicate chimes
                
                # Wind chime synthesis: purer tone with subtle shimmer
                # Chimes are more harmonic than bells, with gentle beating
                vibrato_rate = 4.0 + np.random.random() * 2  # Subtle random vibrato
                vibrato_depth = 0.003  # Very subtle pitch wobble
                vibrato = 1 + vibrato_depth * np.sin(2 * np.pi * vibrato_rate * t_chunk)
                
                # Wind chimes have purer, more harmonic overtones
                tone = scaled_amp * env_chunk * (
                    1.0 * np.sin(2 * np.pi * freq * vibrato * t_chunk) +           # Fundamental with vibrato
                    0.3 * np.sin(2 * np.pi * freq * 2.0 * t_chunk) +               # Soft 2nd harmonic
                    0.15 * np.sin(2 * np.pi * freq * 3.0 * t_chunk) +              # Gentle 3rd
                    0.08 * np.sin(2 * np.pi * freq * 4.0 * t_chunk)                # Faint 4th
                )
                chunk += tone
            
            if np.max(np.abs(chunk)) > 0:
                chunk = chunk / np.max(np.abs(chunk)) * 0.6  # Softer overall
            
            audio[t_start:t_end] = chunk
        
        print(f"   Progress: {n_time_steps}/{n_time_steps} (100.0%)")
        print(f"   Active time steps with sound: {active_count}/{n_time_steps} ({100*active_count/n_time_steps:.1f}%)")
        
        if np.max(np.abs(audio)) > 0:
            audio = audio / np.max(np.abs(audio)) * 0.9
        
        return audio
    
    def _map_to_pentatonic(self, n_bins):
        """Map frequency bins to selected musical scale"""
        # Get the appropriate scale
        if self.mapping_mode in self.SCALES:
            scale_ratios = self.SCALES[self.mapping_mode]
        else:
            scale_ratios = self.SCALES['pentatonic']  # Default fallback
        
        base_freq = MIN_AUDIBLE_FREQ
        n_octaves = int(np.log2(MAX_AUDIBLE_FREQ / MIN_AUDIBLE_FREQ))
        
        freqs = []
        for i in range(n_bins):
            octave = int(i * n_octaves / n_bins)
            note_idx = i % len(scale_ratios)
            freq = base_freq * (2 ** octave) * scale_ratios[note_idx]
            freqs.append(min(freq, MAX_AUDIBLE_FREQ))
        
        return np.array(freqs)
    
    def _map_proportional(self, radio_freqs):
        """Map radio frequencies proportionally to audible range"""
        log_radio = np.log10(radio_freqs + 1)
        log_min, log_max = log_radio.min(), log_radio.max()
        normalized = (log_radio - log_min) / (log_max - log_min + 1e-10)
        
        log_audible_min = np.log10(MIN_AUDIBLE_FREQ)
        log_audible_max = np.log10(MAX_AUDIBLE_FREQ)
        log_audible = log_audible_min + normalized * (log_audible_max - log_audible_min)
        
        return 10 ** log_audible
    
    def save_wav(self, audio, filename, sample_rate=44100):
        """Save audio to WAV file"""
        audio_int16 = (audio * 32767).astype(np.int16)
        wav_write(filename, sample_rate, audio_int16)
        print(f"🎧 Saved: {filename}")
        return filename


def play_spectrogram_with_audio(callisto_obj, mapping_mode='pentatonic', 
                                  audio_duration=15, sample_rate=44100, xtick=2,
                                  save_video=False, video_filename="output.mp4"):
    """Play spectrogram animation with synchronized audio, optionally save as video"""
    import pyCallistoUtils as utils

    print("=" * 60)
    print("SOLAR RADIO SPECTROGRAM SONIFICATION")
    print("Based on e-callisto acoustic spectrum representation")
    print("=" * 60)
    
    # Extract data using pyCallisto structure
    spectrogram_data = callisto_obj.imageHdu.data
    frequencies = callisto_obj.binTableHdu.data['frequency'][0]
    
    # Get time information
    startDate = utils.toDate(callisto_obj.imageHeader['DATE-OBS'])
    startTime = utils.toTime(callisto_obj.imageHeader['TIME-OBS'])
    startDateTime = dt.datetime.combine(startDate, startTime)
    
    endTime = utils.toTime(callisto_obj.imageHeader['TIME-END'])
    endDateTime = dt.datetime.combine(startDate, endTime)
    
    vmax = callisto_obj.dataMax
    
    # Downsample for audio if needed
    target_time_steps = max(200, min(1000, spectrogram_data.shape[1]))
    if spectrogram_data.shape[1] > target_time_steps:
        step = spectrogram_data.shape[1] // target_time_steps
        spectrogram_data_audio = spectrogram_data[:, ::step]
        print(f"   Applying intensity threshold: {INTENSITY_THRESHOLD} (keeping top {100*(1-INTENSITY_THRESHOLD):.0f}% of intensities)")
        print(f"   Downsampled to {spectrogram_data_audio.shape[1]} time steps")
    else:
        spectrogram_data_audio = spectrogram_data
    
    # Generate audio
    print(f"🎵 Generating {audio_duration}s audio with {mapping_mode} mapping...")
    print(f"   Only sonifying features with intensity >= {INTENSITY_THRESHOLD}")
    sonifier = SpectrogramSonifier(mapping_mode=mapping_mode)
    audio = sonifier.generate_audio(spectrogram_data_audio, frequencies, 
                                     duration=audio_duration, sample_rate=sample_rate,
                                     intensity_threshold=INTENSITY_THRESHOLD)
    
    # Mix with custom MP3 if enabled (intensity-gated: MP3 only plays during high density regions)
    if USE_CUSTOM_MP3 and PYDUB_AVAILABLE:
        mp3_audio = load_mp3_as_numpy(CUSTOM_MP3_PATH, audio_duration, sample_rate)
        if mp3_audio is not None:
            audio = mix_audio_intensity_gated(audio, mp3_audio, spectrogram_data_audio,
                                               SONIFICATION_VOLUME, MP3_VOLUME,
                                               INTENSITY_THRESHOLD, sample_rate)
    
    wav_file = "spectrogram(20250225)_audio_output.wav"
    sonifier.save_wav(audio, wav_file, sample_rate)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 6))
    
    xLims = mdates.date2num([startDateTime, endDateTime])
    yLims = [frequencies[-2], frequencies[1]]
    
    cax = ax.imshow(spectrogram_data, 
                    extent=[xLims[0], xLims[1], yLims[0], yLims[1]], 
                    aspect='auto', 
                    cmap=cm.jet, 
                    vmin=-10, 
                    vmax=vmax)
    
    ax.xaxis_date()
    ax.xaxis.set_major_locator(mdates.MinuteLocator(byminute=range(60), interval=xtick, tz=None))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
    fig.autofmt_xdate()
    
    ax.set_xlabel('Universal Time', fontsize=12)
    ax.set_ylabel('Frequency (MHz)', fontsize=12)
    ax.set_title(callisto_obj.imageHeader.get('CONTENT', 'Solar Radio Spectrogram'), fontsize=14)
    
    cbar = fig.colorbar(cax, ax=ax, orientation='vertical')
    cbar.set_label('Intensity', rotation=90)
    
    # Create tracking line
    tracking_line = ax.axvline(x=xLims[0], color='red', linestyle='-', linewidth=2, alpha=0.8)
    
    time_text = ax.text(0.02, 0.98, '', transform=ax.transAxes,
                        verticalalignment='top', fontsize=10,
                        color='white', bbox=dict(boxstyle='round', facecolor='black', alpha=0.5))
    
    start_time = [None]
    audio_process = [None]
    
    def init():
        tracking_line.set_xdata([xLims[0], xLims[0]])
        time_text.set_text('')
        return tracking_line, time_text
    
    def animate(frame):
        import time
        
        if start_time[0] is None:
            start_time[0] = time.time()
            try:
                audio_process[0] = subprocess.Popen(
                    ['aplay', wav_file],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL
                )
            except Exception as e:
                print(f"Could not play audio: {e}")
        
        elapsed = time.time() - start_time[0]
        progress = min(elapsed / audio_duration, 1.0)
        
        current_x = xLims[0] + progress * (xLims[1] - xLims[0])
        tracking_line.set_xdata([current_x, current_x])
        
        current_time = mdates.num2date(current_x)
        time_text.set_text(f'Time: {current_time.strftime("%H:%M:%S")}\nProgress: {progress*100:.1f}%')
        
        if progress >= 1.0:
            if audio_process[0] is not None:
                audio_process[0].terminate()
        
        return tracking_line, time_text
    
    n_time_steps = spectrogram_data.shape[1]
    fps = 30
    total_frames = int(audio_duration * fps)
    
    print(f"\n🎬 Starting animation ({audio_duration}s)")
    print(f"📊 Spectrogram: {n_time_steps} time steps, {len(frequencies)} frequency bins")
    print(f"🎵 Mapping mode: {mapping_mode}")
    
    if save_video:
        # Save video with audio
        print(f"\n📹 Saving video with audio to: {video_filename}")
        print(f"   This may take a few minutes...")
        
        # Create temporary directory for frames
        temp_dir = tempfile.mkdtemp()
        
        try:
            # Animation function for video (no audio playback)
            def animate_for_video(frame_num):
                progress = frame_num / total_frames
                current_x = xLims[0] + progress * (xLims[1] - xLims[0])
                tracking_line.set_xdata([current_x, current_x])
                current_time = mdates.num2date(current_x)
                time_text.set_text(f'Time: {current_time.strftime("%H:%M:%S")}\nProgress: {progress*100:.1f}%')
                return tracking_line, time_text
            
            # Render frames
            print(f"   Rendering {total_frames} frames...")
            for frame_num in range(total_frames):
                if frame_num % 50 == 0:
                    print(f"   Progress: {frame_num}/{total_frames} ({100*frame_num/total_frames:.1f}%)")
                
                animate_for_video(frame_num)
                fig.canvas.draw()
                
                frame_path = os.path.join(temp_dir, f"frame_{frame_num:05d}.png")
                fig.savefig(frame_path, dpi=100, bbox_inches='tight', pad_inches=0.1)
            
            print(f"   Progress: {total_frames}/{total_frames} (100.0%)")
            print(f"   ✓ All frames rendered")
            
            # Use ffmpeg to combine frames and audio
            print(f"   Combining frames and audio...")
            frame_pattern = os.path.join(temp_dir, "frame_%05d.png")
            
            cmd = [
                'ffmpeg', '-y',
                '-framerate', str(fps),
                '-i', frame_pattern,
                '-i', wav_file,
                '-vf', 'scale=trunc(iw/2)*2:trunc(ih/2)*2',
                '-af', 'volume=2.0',
                '-c:v', 'libx264',
                '-preset', 'medium',
                '-crf', '23',
                '-pix_fmt', 'yuv420p',
                '-c:a', 'aac',
                '-b:a', '192k',
                '-shortest',
                video_filename
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            # Cleanup temporary directory
            shutil.rmtree(temp_dir, ignore_errors=True)
            
            if result.returncode == 0:
                print(f"\n✅ Video with audio saved successfully!")
                print(f"   📁 File: {video_filename}")
                file_size = os.path.getsize(video_filename) / (1024 * 1024)
                print(f"   📊 File size: {file_size:.1f} MB")
                print(f"\n   🎐 Wind chimes sonification complete!")
            else:
                print(f"⚠️ FFmpeg error: {result.stderr}")
                
        except Exception as e:
            print(f"⚠️ Error saving video: {e}")
            shutil.rmtree(temp_dir, ignore_errors=True)
    else:
        # Live playback
        ani = FuncAnimation(fig, animate, init_func=init,
                            frames=total_frames, interval=1000/fps,
                            blit=True, repeat=False)
        
        plt.tight_layout()
        plt.show()
        
        # Cleanup
        if audio_process[0] is not None:
            audio_process[0].terminate()
    
    plt.close(fig)
