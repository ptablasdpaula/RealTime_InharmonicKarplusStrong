import numpy as np
from utils import hz_to_samples

class DynamicDelayLine:
    def __init__(self, max_delay: int = 2000):
        """Fixed-size circular buffer - no resizing needed!

        Args:
            max_delay: Maximum delay length in samples (for ~10 Hz at 16kHz)
        """
        self.max_delay = max_delay
        self.buffer = np.zeros(max_delay)
        self.write_idx = 0
        self.delay_length = max_delay // 2  # Start at middle

    def set_delay(self, delay_samples: int):
        """Change delay length without resizing - just move read point!"""
        self.delay_length = max(1, min(delay_samples, self.max_delay - 1))

    @property
    def read_idx(self):
        """Read pointer is always delay_length behind write pointer."""
        return (self.write_idx - self.delay_length) % self.max_delay

    def peek(self) -> float:
        return float(self.buffer[self.read_idx])

    def write(self, sample: float):
        self.buffer[self.write_idx] = sample
        self.write_idx = (self.write_idx + 1) % self.max_delay

    def reset_state(self):
        """Clear buffer contents."""
        self.buffer.fill(0)


class LagrangeFilter:
    def __init__(self, order=5):
        self.order = order
        self.buffer = np.zeros(order + 1)
        self.coeffs = np.zeros(order + 1)
        self.alpha = None

    def _update_coefficients(self, alpha):
        """Compute Lagrange interpolation coefficients using the formula."""
        self.alpha = alpha
        for k in range(self.order + 1):
            num = 1.0
            den = 1.0
            for m in range(self.order + 1):
                if m != k:
                    num *= (alpha - m)
                    den *= (k - m)
            self.coeffs[k] = num / den

    def process_sample(self, sample: float, alpha: float) -> float:
        if alpha != self.alpha: self._update_coefficients(alpha)
        self.buffer[1:] = self.buffer[:-1]
        self.buffer[0] = sample
        output = np.dot(self.coeffs, self.buffer)
        return output

    def reset_state(self):
        self.alpha = None
        self.buffer.fill(0)


def split_centered_delay(delay_length: float, order: int = 5):
    """
    Split delay into integer and fractional parts for centered Lagrange filter.
    Returns: (integer_delay, alpha_for_lagrange)
    """
    z_floor = int(delay_length)
    z_center = z_floor - (order // 2)
    alpha = delay_length - z_center
    return z_center, alpha


class IIR1stODCNormFilter:
    def __init__(self, sr: int = 16000):
        self.sr = sr
        self.y_prev = 0.0

    def process_sample(self, sample: float, a1: float) -> float:
        b0 = 1.0 - a1
        output = b0 * sample + a1 * self.y_prev
        self.y_prev = output
        return output

    def phase_delay_at_f0(self, f0: float, a1: float) -> float:
        omega = 2 * np.pi * f0 / self.sr
        b0 = 1.0 - a1

        # Evaluate at e^(jω)
        ejw = np.exp(1j * omega)
        H = b0 / (1 - a1 / ejw)

        # Phase delay: -phase(H) / ω
        phase = np.angle(H)
        phase_delay = -phase / omega

        return phase_delay

    def reset_state(self):
        self.y_prev = 0.0


class AllpassBiquad:
    """
    Single second-order allpass filter section.

    Transfer function: H(z) = (a² + 2ac·z⁻¹ + z⁻²) / (1 + 2ac·z⁻¹ + a²·z⁻²)
    where a = pole radius, c = cos(pole angle)
    """

    def __init__(self, sr: int = 16000):
        self.sr = sr
        self.state = np.zeros(4)  # [x1, x2, y1, y2]
        self.coeffs = np.zeros(6)  # [b0, b1, b2, a0, a1, a2]
        self.set_to_passthrough()

    def set_to_passthrough(self):
        self.coeffs = np.array([1.0, 0.0, 0.0, 1.0, 0.0, 0.0])

    def design_from_pole(self, pole_radius: float, pole_cosine: float):
        a = pole_radius
        c = pole_cosine
        self.coeffs = np.array([
            a ** 2,  # b0
            2 * a * c,  # b1
            1.0,  # b2
            1.0,  # a0
            2 * a * c,  # a1
            a ** 2  # a2
        ])

    def process_sample(self, sample: float) -> float:
        b0, b1, b2, a0, a1, a2 = self.coeffs
        x1, x2, y1, y2 = self.state

        # Biquad difference equation
        # y[n] = (b0*x[n] + b1*x[n-1] + b2*x[n-2] - a1*y[n-1] - a2*y[n-2]) / a0
        y = (b0 * sample + b1 * x1 + b2 * x2 - a1 * y1 - a2 * y2) / a0
        self.state = np.array([sample, x1, y, y1])
        return y

    def reset_state(self):
        self.state = np.zeros(4)

    def phase_delay_at_f0(self, f0: float) -> float:
        """Compute phase delay at frequency f0 in samples."""
        omega = 2 * np.pi * f0 / self.sr

        b0, b1, b2, a0, a1, a2 = self.coeffs

        # Evaluate H(z) at z = e^(jω)
        ejw = np.exp(1j * omega)
        ejw_inv = 1.0 / ejw
        ejw_inv2 = ejw_inv * ejw_inv

        # H(z) = (b0 + b1*z^-1 + b2*z^-2) / (a0 + a1*z^-1 + a2*z^-2)
        num = b0 + b1 * ejw_inv + b2 * ejw_inv2
        den = a0 + a1 * ejw_inv + a2 * ejw_inv2

        H = num / den

        # Phase delay: τ_p(ω) = -φ(ω) / ω
        phase = np.angle(H)
        phase_delay = -phase / omega

        return phase_delay


class AllpassDispersionFilterCascade:
    """
    Cascaded allpass filter for modeling string dispersion (inharmonicity).
    Based on Abel, Välimäki, Smith (2010) IEEE SPL paper.
    """

    def __init__(self, max_sections=12, sr=16000, smoothing=0.85):
        """
        Initialize dispersion filter cascade.

        Args:
            max_sections: Maximum number of biquad sections (fixed for real-time)
            sr: Sampling rate in Hz
            smoothing: Smoothing factor (0.75-0.85 recommended)
        """
        self.max_sections = max_sections
        self.sr = sr
        self.smoothing = smoothing

        # Create array of independent biquad sections
        self.sections = [AllpassBiquad(sr=sr) for _ in range(max_sections)]

        self.last_f0 = None
        self.last_inharmonicity = None
        self.active_sections = 0

    def _set_inactive_to_passthrough(self):
        """Set unused sections to passthrough."""
        for i in range(self.active_sections, self.max_sections):
            self.sections[i].set_to_passthrough()

    def _design_allpass_sections(self, pole_radius: np.ndarray, pole_cosine: np.ndarray):
        """Design active sections from pole locations."""
        for i in range(self.active_sections):
            self.sections[i].design_from_pole(
                float(pole_radius[i]),
                float(pole_cosine[i]))

    @staticmethod
    def _scale_bandwidth(f0: float, inharmonicity: float) -> float:
        # Base bandwidth (for typical strings)
        if f0 < 100:
            base_bw = 2000.0
        elif f0 < 200:
            base_bw = 3000.0
        else:
            base_bw = 4000.0

        # Scale up for extreme inharmonicity
        if inharmonicity > 0.001:
            # Scale factor: 1x at B=0.001, up to 2x at B≥0.01
            scale_factor = min(2.0, inharmonicity / 0.001)
            return base_bw * (1.0 + scale_factor * 0.5)

        return base_bw

    def _find_pole_frequencies(self, f0: float, inharmonicity: float,
                               design_bandwidth: float, group_delay_0: float) -> np.ndarray:
        """
        Compute pole frequencies using Newton iteration.
        Returns frequencies where phase equals target phases at odd multiples of π (-π, -3π, -5π, ...).
        """
        target_phases = np.pi * np.arange(1, 2 * self.active_sections, 2)

        # Initial frequency estimates (linearly spaced, biased toward DC)
        initial_pole_f = np.arange(self.active_sections) / (1.2 * self.active_sections) * design_bandwidth

        # Compute phase and group delay at initial frequencies
        f0_samples = hz_to_samples(f0, self.sr)
        phase_delay_f = f0_samples / np.sqrt(1 + inharmonicity * (initial_pole_f / f0) ** 2)
        group_delay_f = phase_delay_f / (1 + inharmonicity * (initial_pole_f / f0) ** 2)
        phase_f = 2 * np.pi * (initial_pole_f / self.sr) * phase_delay_f

        # Newton iteration: refine frequencies to match target phases
        pole_f = self.sr / (2 * np.pi) * \
                 (target_phases - phase_f + (2 * np.pi / self.sr) * initial_pole_f * group_delay_f) / \
                 (group_delay_f - group_delay_0)

        return pole_f

    def _compute_pole_radii(self, pole_f: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Pole radius determined by spacing between adjacent poles and smoothing factor smoothing.
        Closer poles → smaller radius (less resonance).
        """
        pole_half_bandwidth = np.diff(np.concatenate([[-pole_f[0]], pole_f])) / 2
        pole_cosine = np.cos(pole_f * 2 * np.pi / self.sr)
        norm_factor = (1 - self.smoothing * np.cos(pole_half_bandwidth * 2 * np.pi / self.sr)) / (1 - self.smoothing)
        pole_radius = np.sqrt(norm_factor ** 2 - 1) - norm_factor
        return pole_radius, pole_cosine

    def _update_coefficients(self, f0: float, inharmonicity: float):
        design_bandwidth = self._scale_bandwidth(f0, inharmonicity)
        f0_samples = hz_to_samples(f0, self.sr)

        # Compute delay characteristics at design bandwidth
        phase_delay_0 = f0_samples / np.sqrt(1 + inharmonicity * (design_bandwidth / f0) ** 2)
        group_delay_0 = phase_delay_0 / (1 + inharmonicity * (design_bandwidth / f0) ** 2)

        # Determine number of sections needed
        integrated_phase_0 = 2 * np.pi * (design_bandwidth / self.sr) * phase_delay_0 - \
                             group_delay_0 * 2 * np.pi * design_bandwidth / self.sr
        min_sections = int(np.floor(integrated_phase_0 / (2 * np.pi)))
        self.active_sections = min(min_sections, self.max_sections)

        if self.active_sections <= 0:
            self.active_sections = 0
            self._set_inactive_to_passthrough()
            return

        # Design pole locations
        pole_frequencies = self._find_pole_frequencies(f0, inharmonicity, design_bandwidth, group_delay_0)
        pole_radius, pole_cosine = self._compute_pole_radii(pole_frequencies)

        # Update biquad sections
        self._design_allpass_sections(pole_radius, pole_cosine)
        self._set_inactive_to_passthrough()

    def _params_changed(self, f0: float, inharmonicity: float):
        if self.last_f0 != f0 or self.last_inharmonicity != inharmonicity:
            self._update_coefficients(f0, inharmonicity)
            self.last_f0 = f0
            self.last_inharmonicity = inharmonicity

    def process_sample(self, sample: float, f0: float, inharmonicity: float) -> float:
        self._params_changed(f0, inharmonicity)
        output = sample
        for i in range(self.active_sections):
            output = self.sections[i].process_sample(output)
        return output

    def phase_delay_at_f0(self, f0: float) -> float:
        if self.active_sections == 0:
            return 0.0

        total_phase_delay = 0.0
        for i in range(self.active_sections):
            total_phase_delay += self.sections[i].phase_delay_at_f0(f0)

        return total_phase_delay

    def reset_state(self):
        for section in self.sections:
            section.reset_state()


class InharmonicKarplusStrong:
    def __init__(self, sr: int = 16000, max_dispersion_sections: int = 20, lagrange_order: int = 5):
        self.sr = sr

        # Fixed-size delay line (max_delay for ~10 Hz at 16kHz)
        self.delay = DynamicDelayLine(max_delay=2000)
        self.lagrange = LagrangeFilter(order=lagrange_order)
        self.iir_filter = IIR1stODCNormFilter(sr=sr)
        self.dispersion = AllpassDispersionFilterCascade(sr=sr, max_sections=max_dispersion_sections)

        self.lagrange_order = lagrange_order
        self.last_f0 = None
        self.last_inharmonicity = None
        self.last_damping = None
        self.tuned_f0 = None
        self.effective_inharmonicity = 0.0

    def _params_changed(self, f0: float, inharmonicity: float, damping: float):
        """Check if parameters changed and update accordingly."""
        if (f0 != self.last_f0 or
                inharmonicity != self.last_inharmonicity or
                damping != self.last_damping):
            self._update_delay_length(f0, inharmonicity, damping)
            self._reset_state()
            self.last_f0 = f0
            self.last_inharmonicity = inharmonicity
            self.last_damping = damping

    def process_sample(self, excitation: float, f0: float, decay: float,
                       damping: float, inharmonicity: float = 0.0) -> float:
        self._params_changed(f0, inharmonicity, damping)

        delayed = self.delay.peek()
        lagrange_out = self.lagrange.process_sample(delayed, self.alpha)
        dispersed = self.dispersion.process_sample(lagrange_out, self.tuned_f0, self.effective_inharmonicity)
        filtered = self.iir_filter.process_sample(dispersed, damping)
        to_write = excitation + filtered * decay
        self.delay.write(to_write)
        return filtered

    def _reset_state(self):
        self.lagrange.reset_state()
        self.iir_filter.reset_state()
        self.dispersion.reset_state()

    def _update_delay_length(self, f0: float, inharmonicity: float, damping: float):
        """Compute tuned delay length compensating for IIR and dispersion phase delay."""
        f0_samples = hz_to_samples(f0, self.sr)

        # Minimum safe delay AFTER centering (absolute minimum for delay line)
        min_safe_integer_delay = 1

        # Try with requested inharmonicity, reduce if needed
        safe_inharmonicity = inharmonicity
        max_iterations = 20

        for iteration in range(max_iterations):
            # Design dispersion with current inharmonicity
            self.dispersion._update_coefficients(f0, safe_inharmonicity)

            # Compute phase delays at f0 (in samples)
            iir_phase_delay = self.iir_filter.phase_delay_at_f0(f0, damping)
            dispersion_phase_delay = self.dispersion.phase_delay_at_f0(f0)
            total_phase_delay = iir_phase_delay + dispersion_phase_delay

            # Compute compensated delay
            tuned_delay_samples = f0_samples - total_phase_delay

            # Split into integer + fractional
            integer_delay, alpha = split_centered_delay(tuned_delay_samples, order=self.lagrange_order)

            # Check AFTER split - this is the critical check!
            if integer_delay >= min_safe_integer_delay:
                # Success! Store alpha and break
                self.alpha = alpha
                break

            # Reduce inharmonicity and try again
            safe_inharmonicity *= 0.9

            if iteration == max_iterations - 1:
                # Last resort: set to zero
                safe_inharmonicity = 0.0
                self.dispersion._update_coefficients(f0, 0.0)
                iir_phase_delay = self.iir_filter.phase_delay_at_f0(f0, damping)
                tuned_delay_samples = f0_samples - iir_phase_delay
                integer_delay, self.alpha = split_centered_delay(tuned_delay_samples, order=self.lagrange_order)
                print(f"Warning: Inharmonicity too high for f0={f0:.2f}Hz. Setting B=0 (harmonic)")

        # If we reduced inharmonicity significantly, inform user once
        if abs(safe_inharmonicity - inharmonicity) > 0.0001:
            print(f"Note: Clamped B from {inharmonicity:.6f} to {safe_inharmonicity:.6f} for f0={f0:.2f}Hz")

        # Set delay length (no resize, just change read position!)
        self.delay.set_delay(integer_delay)

        # Store f0 and effective inharmonicity
        self.tuned_f0 = f0
        self.effective_inharmonicity = safe_inharmonicity