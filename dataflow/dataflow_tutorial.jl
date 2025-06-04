## Load required Julia packages:

using LinearAlgebra, Random
using SolidStateDetectors
using Geant4: G4JLApplication
using RadiationDetectorSignals
using RadiationDetectorDSP
using RadiationSpectra
using Unitful, UnitfulAtomic
using InverseFunctions
using StatsBase
using Plots
using LazyReports
using JLD2



# Simulating detector with fields 

function build_detsim(::Type{T}, example_geometry::Symbol) where T<:Real
    detector_config_filename = SSD_examples[example_geometry]
    detsim = Simulation{T}(detector_config_filename)
    calculate_electric_potential!(detsim, convergence_limit = 1e-6, refinement_limits = [0.2, 0.1, 0.05, 0.01])
    calculate_electric_field!(detsim, n_points_in_φ = 72)
    for contact in detsim.detector.contacts
        calculate_weighting_potential!(detsim, contact.id, refinement_limits = [0.2, 0.1, 0.05, 0.01], n_points_in_φ = 2, verbose = false)
    end
    charge_drift_model = ADLChargeDriftModel()
    detsim.detector = SolidStateDetector(detsim.detector, charge_drift_model);
    return detsim
end

detsim = build_detsim(Float32, :InvertedCoax)
JLD2.save("snapshots/detsim.jld2", "obj", detsim)


# Plot detector geometry
det_plt = plot(detsim.detector, show_passives = false, size = (700, 700), fmt = :png)

# Plot electric field and field lines
field_plt = plot(detsim.electric_field, full_det = true, φ = 0.0, size = (700, 700),
    xunit = u"mm", yunit = u"mm", zunit = u"V/mm", clims = (0,800).*u"V/mm")
plot_electric_fieldlines!(field_plt, detsim, full_det = true, φ = 0.0)


# Placing a source and generating MC mctruth with Geant4
radsource = MonoenergeticSource(
    "gamma",                       # Type of particle beam
    2.615u"MeV",                   # Energy of particle
    CartesianPoint(0., 0., 0.12),  # Location of the source
    CartesianVector(0,0,-1),       # Direction of the source
    40u"°"                         # Opening angle of the source emission
)

det_src_plt = plot!(deepcopy(det_plt), radsource)

g4app = G4JLApplication(detsim, radsource, verbose = false);
N_events = 30000
mctruth = run_geant4_simulation(g4app, N_events)


# Plot generated hits

det_src_evts_plt = plot!(
    deepcopy(det_src_plt),
    CartesianPoint.(broadcast(p -> ustrip.(u"m", p), mctruth[1:1000].pos.data)),
    ms = 0.5, msw = 0, color=:black, label = ""
)


# Many hits in each event:

mctruth.edep isa AbstractVector{<:AbstractVector{<:Unitful.RealOrRealQuantity}}

mctruth_spec_plt = stephist(
    ustrip.(u"keV", sum.(mctruth.edep)),
    nbins = 0:2:2800, normalize = :density, yscale = :log10,
    label = "mctruth", xlabel = "E_dep / keV", ylabel = "counts / keV",
)


# Simulating waveforms

sel_mctruth = mctruth[begin:begin+1000-1]
wf_gen_tbl = simulate_waveforms(sel_mctruth, detsim, Δt = 1u"ns", max_nsteps = 2000)
sim_wfs = add_baseline_and_extend_tail.(wf_gen_tbl.waveform, 10000, 20000)


# DAQ response

function add_noise(waveform::RDWaveform, noise_level::Real)
    (;signal, time) = waveform
    newsig = ustrip.(signal)
    for i in eachindex(newsig)
        newsig[i] += randn() * noise_level
    end
    return RDWaveform(time, newsig)
end

noisy_wfs = add_noise.(sim_wfs, 20000)

plot(noisy_wfs[1:20], label = "")

amplifier_response = RCFilter(rc = 50u"ns") ∘ CRFilter(20u"μs")
measured_wfs = amplifier_response.(noisy_wfs)

meas_wfs_plt = plot(measured_wfs[1:20], label = "")


# Energy reconstruction via waveform DSP

reco_filter = TrapezoidalChargeFilter(avgtime = 5u"μs", gaptime = 1u"μs") ∘ inverse(CRFilter(20u"μs"))
reco_flt_wfs = reco_filter.(measured_wfs)
E_dsp = maximum.(reco_flt_wfs.signal)


# Uncalibrated measured spectrum

hist_uncal = fit(Histogram, filter(x -> x > 1000, E_dsp), nbins = 1000)
plot(hist_uncal, linetype = :stepbins)


# Calibrated measured spectrum

gamma_lines = [2614.533 - 2*511, 2614.533 - 511, 2614.533] 
hist_cal, _ = RadiationSpectra.calibrate_spectrum(hist_uncal, gamma_lines)

truth_reco_spec_plt = plot!(
    mctruth_spec_plt,
    normalize(hist_cal, mode = :density), linetype = :stepbins, yscale = :log10,
    label = "reco", xlabel = "E_dep / keV", ylabel = "counts / keV",
)


# Generate a report

rpt = lazyreport(
    "# Simulation and reconstruction report",

    "## Detector with source and events",
    det_src_evts_plt,

    "## Detector E-field",
    field_plt,

    "## Measured detector waveforms",
    meas_wfs_plt,
    "(Selection of events shown.)",

    "## Sim-Truth and reconstructed spectrum",
    truth_reco_spec_plt,
    "Note: only a fraction of the MC output was fully simulated and reconstructed.",
)

display(rpt)

write_lazyreport("report.html", rpt)
