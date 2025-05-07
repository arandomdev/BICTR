import random
from typing import cast

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import scipy.signal  # type: ignore
from matplotlib.axes import Axes

import lwchm.model
import lwchm.signal
import lwchm.spatial


def demodulateQPSK(
    signal: npt.NDArray[np.complex128],
    carrier_freq: float,
    symbol_rate: float,
    sample_rate: float,
    isModulated: bool,
) -> npt.NDArray[np.complex128]:
    samples_per_symbol = int(sample_rate / symbol_rate)
    num_symbols = len(signal) // samples_per_symbol
    iq_points: list[np.complex128] = []

    for i in range(num_symbols):
        start = i * samples_per_symbol
        end = start + samples_per_symbol
        t = np.arange(start, end) / sample_rate

        if isModulated:
            # Mix with cosine and sine
            i_component = signal[start:end] * np.cos(2 * np.pi * carrier_freq * t)
            q_component = signal[start:end] * -np.sin(2 * np.pi * carrier_freq * t)

            # Integrate (sum)
            i_sum = np.mean(i_component)
            q_sum = np.mean(q_component)

            iq_points.append(i_sum + 1j * q_sum)
        else:
            iq_points.append(np.mean(signal[start:end]))

    return np.array(iq_points)


def plot_constellation(ax: Axes, iq: npt.NDArray[np.complex128], title: str):
    ax.scatter(iq.real, iq.imag, color="blue")  # type: ignore
    ax.axhline(0, color="gray", linestyle="--", linewidth=0.5)  # type: ignore
    ax.axvline(0, color="gray", linestyle="--", linewidth=0.5)  # type: ignore
    ax.grid(True)  # type: ignore
    ax.set_title(title)  # type: ignore
    ax.set_xlabel("In-phase (I)")  # type: ignore
    ax.set_ylabel("Quadrature (Q)")  # type: ignore
    ax.axis("equal")


def exp3_4() -> None:
    passbandFs = 3e9
    basebandFs = 300e6
    carrFs = 1000e6
    symbolRate = 3e6

    # Generate TX signals
    random.seed(0)
    data = random.randbytes(32)
    passbandConf = lwchm.signal.PSKConfiguration(
        data, 1 / symbolRate, passbandFs, carrFs, 30, True
    )
    basebandConf = lwchm.signal.PSKConfiguration(
        data, 1 / symbolRate, basebandFs, carrFs, 30, False
    )
    passbandTx = lwchm.signal.generateQPSKSignal(passbandConf)
    basebandTx = lwchm.signal.generateQPSKSignal(basebandConf)

    # Load channel
    modelConf = lwchm.model.LWCHMConfiguration(
        refCount=5,
        refAttemptPerRing=10,
        ringRadiusMin=5,
        ringRadiusMax=300,
        ringRadiusUncertainty=15,
        ringCount=10,
        complexRelPermittivityReal=7.058396,
        complexRelPermittivityRealStd=0.007131,
        complexRelPermittivityImag=-0.862227,
        complexRelPermittivityImagStd=0.001397,
        horizontalPolarization=False,
        fadingPaths=1024,
        fadingDopplerSpread=1,
    )
    body = lwchm.spatial.Body(
        "earth",
        "01s",
        lwchm.spatial.PointGeo(-111.655615, 35.568169),
        lwchm.spatial.PointGeo(-111.610698, 35.613086),
    )
    model = lwchm.model.LWCHM(body, modelConf)
    txPoint = lwchm.spatial.PointGeo(-111.633156, 35.590627)
    rxPoint = lwchm.spatial.PointGeo(-111.628157, 35.592453)

    # Pass signals through channel
    random.seed(1)
    passbandAttenuated = model.compute(txPoint, rxPoint, 10, 2, passbandTx)
    random.seed(1)
    basebandAttenuated = model.compute(txPoint, rxPoint, 10, 2, basebandTx)

    assert passbandAttenuated is not None
    assert basebandAttenuated is not None

    # Demodulate
    basebandRx = demodulateQPSK(
        basebandAttenuated.wave, carrFs, symbolRate, basebandFs, isModulated=False
    )
    # basebandRx *= np.exp(1j * 0.49373977287412596)
    passbandRx = demodulateQPSK(
        passbandAttenuated.wave, carrFs, symbolRate, passbandFs, isModulated=True
    )

    # Convert to symbols
    basebandRxSymbols = np.angle(basebandRx)
    passbandRxSymbols = np.angle(passbandRx)
    nSymbols = min(len(basebandRxSymbols), len(passbandRxSymbols))
    basebandRxSymbols.resize(nSymbols)
    passbandRxSymbols.resize(nSymbols)

    phaseDiff = np.mean(np.abs(basebandRxSymbols - passbandRxSymbols))

    # Print results
    print(
        f"Average abs phase difference: {phaseDiff:>.2f} ({np.rad2deg(phaseDiff):>.2f} Deg)"
    )
    print(
        f"Baseband Attenuated Power: {lwchm.signal.computeRmsDBM(basebandAttenuated.wave):>.2f}"
    )
    print(
        f"Passband Attenuated Power: {lwchm.signal.computeRmsDBM(passbandAttenuated.wave):>.2f}"
    )

    # Plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4))  # type: ignore
    plot_constellation(ax1, basebandRx, "Baseband Rx")
    plot_constellation(ax2, passbandRx, "Passband Rx")

    fig.show()
    plt.show()  # type: ignore
    pass


def exp1_2() -> None:
    passbandFs = 3e9
    basebandFs = 1500e6
    carrFs = 1000e6
    symbolRate = 3e6

    # Generate TX signals
    random.seed(1)
    data = random.randbytes(32)
    passbandConf = lwchm.signal.PSKConfiguration(
        data, 1 / symbolRate, passbandFs, carrFs, 30, True
    )
    basebandConf = lwchm.signal.PSKConfiguration(
        data, 1 / symbolRate, basebandFs, carrFs, 30, False
    )
    passbandTx = lwchm.signal.generateQPSKSignal(passbandConf)
    basebandTx = lwchm.signal.generateQPSKSignal(basebandConf)

    # Define channels
    passbandChannel = np.array(
        [
            8.76299707e-06 + 3.93074296e-07j,
            *np.zeros(26),
            -6.12759263e-08 + 2.91173445e-07j,
            *np.zeros(9),
            4.88663235e-07 - 1.36005638e-06j,
            *np.zeros(11),
            2.37403031e-06 - 1.24061697e-06j,
            *np.zeros(43),
            -8.29126054e-07 - 3.39416591e-06j,
            *np.zeros(152),
            -1.09729143e-06 - 3.57355043e-06j,
        ],
        dtype=np.complex128,
    )
    basebandChannel = cast(
        npt.NDArray[np.complex128],
        scipy.signal.decimate(passbandChannel, 2),  # type: ignore
    )
    basebandChannel *= np.exp(
        -1j * 2 * np.pi * carrFs * np.arange(0, len(basebandChannel)) / basebandFs
    )

    # Pass signal through channels
    basebandAttenuated = cast(
        npt.NDArray[np.complex128],
        scipy.signal.convolve(basebandTx.wave, basebandChannel),  # type: ignore
    )
    passbandAttenuated = cast(
        npt.NDArray[np.complex128],
        scipy.signal.convolve(passbandTx.wave, passbandChannel),  # type: ignore
    )

    # Demodulate
    basebandRx = demodulateQPSK(
        basebandAttenuated, carrFs, symbolRate, basebandFs, isModulated=False
    )
    passbandRx = demodulateQPSK(
        passbandAttenuated, carrFs, symbolRate, passbandFs, isModulated=True
    )

    # Convert to symbols
    basebandRxSymbols = np.angle(basebandRx)
    passbandRxSymbols = np.angle(passbandRx)
    nSymbols = min(len(basebandRxSymbols), len(passbandRxSymbols))
    basebandRxSymbols.resize(nSymbols)
    passbandRxSymbols.resize(nSymbols)

    phaseDiff = np.mean(np.abs(basebandRxSymbols - passbandRxSymbols))

    # Print results
    print(
        f"Average abs phase difference: {phaseDiff:>.2f} ({np.rad2deg(phaseDiff):>.2f} Deg)"
    )

    print(
        f"Baseband Attenuated Power: {lwchm.signal.computeRmsDBM(basebandAttenuated):>.2f}"
    )
    print(
        f"Passband Attenuated Power: {lwchm.signal.computeRmsDBM(passbandAttenuated):>.2f}"
    )

    # Plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4))  # type: ignore
    plot_constellation(ax1, basebandRx, "Baseband Rx")
    plot_constellation(ax2, passbandRx, "Passband Rx")

    fig.show()
    plt.show()  # type: ignore
    pass


def generateTest() -> None:
    symbols = [
        np.pi / 4,  # 00
        3 * np.pi / 4,  # 01
        5 * np.pi / 4,  # 10
        7 * np.pi / 4,  # 11
    ]

    samplesPerBit = 100
    phasesUpsampled = np.repeat(symbols, samplesPerBit)

    # Modulate
    t = np.arange(0, len(phasesUpsampled)) / 10000
    signal1 = np.cos((2 * np.pi * 100 * t) + phasesUpsampled)
    signal2 = np.real(np.exp(1j * phasesUpsampled) * np.exp(1j * 2 * np.pi * 100 * t))

    fig, (ax1, ax2) = plt.subplots(1, 2)
    ax1.plot(t, signal1)
    ax2.plot(t, signal2)
    fig.show()
    plt.show()

    print(all(np.isclose(signal1, signal2)))
    pass


if __name__ == "__main__":
    # exp1_2()
    exp3_4()
    # generateTest()
