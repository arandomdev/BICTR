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
    carrier_freq: float = 1000,
    symbol_rate: float = 100,
    sample_rate: float = 100000,
    isModulated: bool = True,
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
            i_sum = np.sum(i_component)
            q_sum = np.sum(q_component)

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


def main2() -> None:
    passbandFs = 6e9
    basebandFs = 3e6 * 2
    carrFs = 913e6
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
    rxPoint = lwchm.spatial.PointGeo(-111.625833, 35.596667)

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

    # Print results
    print("BasebandTx: ", np.round(np.angle(basebandTx.wave), 4))
    print("BasebandRx: ", np.round(basebandRxSymbols, 4))
    print("PassbandRx: ", np.round(passbandRxSymbols, 4))
    print(
        "Average abs phase difference: ",
        np.mean(np.abs(passbandRxSymbols - basebandRxSymbols)),
    )

    print(
        "Baseband Attenuated Power: ",
        lwchm.signal.computeRmsDBM(basebandAttenuated.wave),
    )
    print(
        "Passband Attenuated Power: ",
        lwchm.signal.computeRmsDBM(passbandAttenuated.wave),
    )

    # Plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4))  # type: ignore
    plot_constellation(ax1, basebandRx, "Baseband Rx")
    plot_constellation(ax2, passbandRx, "Passband Rx")

    fig.show()
    plt.show()  # type: ignore
    pass


def main() -> None:
    passbandFs = 3e9
    basebandFs = 61.44e6
    carrFs = 913e6
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
            2.07200033e-06 - 1.89501364e-06j,
            *np.repeat(0j, 16),
            -1.18638209e-06 + 1.27110832e-06j,
            *np.repeat(0j, 7),
            -2.42142622e-07 + 9.95226135e-07j,
            *np.repeat(0j, 632),
            -1.59581632e-06 - 2.95844218e-07j,
        ],
        dtype=np.complex128,
    )
    basebandChannel = np.array(
        [
            (
                (-1.53304257e-07 - 2.17792856e-06j)
                * np.exp(-1j * 2 * np.pi * carrFs * 0)
                + (2.22530459e-06 + 2.82914923e-07j)
                * np.exp(-1j * 2 * np.pi * carrFs * 1.3847234358495478e-10)
                + (-1.18638209e-06 + 1.27110832e-06j)
                * np.exp(-1j * 2 * np.pi * carrFs * 5.655011854659811e-09)
            ),
            (-2.42142622e-07 + 9.95226135e-07j)
            * np.exp(-1j * 2 * np.pi * carrFs * 8.51171603491485e-09),
            *np.repeat(0j, 11),
            (-1.59581632e-06 - 2.95844218e-07j)
            * np.exp(-1j * 2 * np.pi * carrFs * 2.194243411931998e-07),
        ],
        dtype=np.complex128,
    )

    # passbandChannel = np.array(
    #     [
    #         1,
    #         *np.zeros(int(round((1 / basebandFs) * passbandFs)) - 1),
    #         1,
    #     ],
    #     dtype=np.complex128,
    # )
    # basebandChannel = np.array(
    #     [
    #         1,
    #         1 * np.exp(-1j * 2 * np.pi * carrFs * (1 / basebandFs)),
    #     ],
    #     dtype=np.complex128,
    # )
    # basebandChannel *= np.exp(
    #     -1j * 2 * np.pi * carrFs * np.arange(0, len(basebandChannel)) / basebandFs
    # )

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

    # Print results
    print("BasebandTx: ", np.round(np.angle(basebandTx.wave), 4))
    print("BasebandRx: ", np.round(basebandRxSymbols, 4))
    print("PassbandRx: ", np.round(passbandRxSymbols, 4))
    print("Average phase difference: ", np.std(passbandRxSymbols - basebandRxSymbols))

    print("Baseband Attenuated Power: ", lwchm.signal.computeRmsDBM(basebandAttenuated))
    print("Passband Attenuated Power: ", lwchm.signal.computeRmsDBM(passbandAttenuated))

    # Plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4))  # type: ignore
    plot_constellation(ax1, basebandRx, "Baseband Rx")
    plot_constellation(ax2, passbandRx, "Passband Rx")

    fig.show()
    plt.show()  # type: ignore
    pass


if __name__ == "__main__":
    main()
