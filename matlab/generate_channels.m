function generate_channels(nFrames)
% GENERATE_CHANNELS  Produce ground-truth OFDM channel grids for VehA and SUI5.
%
%   generate_channels           uses the full dataset (40000 frames/model)
%   generate_channels(1000)     quick test with fewer frames
%
% Each channel is a 72 (subcarrier) x 14 (OFDM symbol) complex grid sampled
% at the centre of every symbol. The perfect channel is the CNN target; all
% pilots, noise, LS/MMSE and the networks are handled later in Python.
%
% Output (./data):
%   H_VehA.mat, H_SUI5.mat  with single arrays Hr, Hi of size [72 x 14 x N]
%                           and a cfg struct recording the parameters.
%
% Requires the Vienna 5G Link-Level Simulator unzipped into the project root
% so that  <root>/5GLLSimulator/+Channel/FastFading.m  exists.
% Run from anywhere; the path is resolved from this file's location.
% -------------------------------------------------------------------------

    if nargin < 1 || isempty(nFrames)
        nFrames = 40000;                 % full dataset (paper: 32k/4k/4k)
    end

    % ----- locate the simulator -----------------------------------------
    here = fileparts(mfilename('fullpath'));
    root = fileparts(here);              % project root (parent of matlab/)
    sim  = fullfile(root, '5GLLSimulator');
    if exist(fullfile(sim, '5GLLSimulator', '+Channel'), 'dir')
        sim = fullfile(sim, '5GLLSimulator');     % handle doubly-nested unzip
    end
    assert(exist(fullfile(sim, '+Channel', 'FastFading.m'), 'file') == 2, ...
        'FastFading.m not found under %s', sim);
    addpath(sim);
    outDir = fullfile(root, 'data');
    if ~exist(outDir, 'dir'); mkdir(outDir); end

    % ----- common configuration -----------------------------------------
    cfg.nSub   = 72;                     % subcarriers (6 RB x 12, LTE 1.4 MHz)
    cfg.nSym   = 14;                     % symbols per subframe (normal CP)
    cfg.df     = 15e3;                   % subcarrier spacing [Hz]
    cfg.nFFT   = 256;                    % -> 3.84 MHz sampling rate
    cfg.fs     = cfg.df * cfg.nFFT;
    cfg.nCP    = 18;                     % cyclic prefix samples (~4.69 us)
    cfg.fc     = 2.1e9;                  % carrier [Hz]
    cfg.vKmh   = 50;                     % UE speed [km/h]
    cfg.fD     = cfg.vKmh/3.6 * cfg.fc / physconst('LightSpeed');  % ~97 Hz
    cfg.nPaths = 100;                    % sinusoids per tap (Jakes)
    cfg.sub    = 4;                      % internal time sub-sampling
    cfg.nFrames = nFrames;

    symLen  = cfg.nFFT + cfg.nCP;        % samples per OFDM symbol
    nSamp   = cfg.nSym * symLen;         % samples per subframe
    cfg.sampleTimes = (0:cfg.nSym-1)*symLen + cfg.nCP + cfg.nFFT/2;  % symbol centres
    cfg.activeSub   = 1:cfg.nSub;

    % ----- SUI-5 power-delay profile (not built into the simulator) ------
    % omni, delays 0/4/10 us, powers 0/-5/-10 dB -> linear power per sample
    suiDelay = [0 4e-6 10e-6];
    suiPdB   = [0 -5 -10];
    tap      = round(suiDelay * cfg.fs) + 1;         % samples 1, 16, 39
    suiPdp          = zeros(1, max(tap));
    suiPdp(tap)     = 10.^(suiPdB/10);

    models = struct( ...
        'name', {'VehA',        'SUI5'}, ...
        'pdp',  {'VehicularA',  suiPdp}, ...
        'seed', {1,             2});

    % ----- generation ----------------------------------------------------
    for mi = 1:numel(models)
        name = models(mi).name;
        rng(models(mi).seed);

        ch = Channel.FastFading( ...
            cfg.fs, models(mi).pdp, nSamp, cfg.fD, 'Jakes', cfg.nPaths, ...
            false, 'none', 0, 0, 1, 1, false, 0, 0, ...
            struct('spatialChannelModel', false), ...
            [1 1], [1 1], cfg.fc, cfg.sub);

        fprintf('%s: RMS delay spread = %.2f us\n', ...
                name, ch.GetRmsDelaySpread*1e6);

        Hr = zeros(cfg.nSub, cfg.nSym, nFrames, 'single');
        Hi = zeros(cfg.nSub, cfg.nSym, nFrames, 'single');
        tic;
        for n = 1:nFrames
            ch.NewRealization(n);
            Hf = ch.GetTransferFunction(cfg.sampleTimes, cfg.nFFT, cfg.activeSub);
            Hr(:,:,n) = single(real(Hf));
            Hi(:,:,n) = single(imag(Hf));
            if mod(n, 2000) == 0
                fprintf('  %s  %d / %d   (%.0f s)\n', name, n, nFrames, toc);
            end
        end
        meanPow = mean(Hr(:).^2 + Hi(:).^2);
        fprintf('%s done (%.0f s), mean |H|^2 = %.3f\n', name, toc, meanPow);

        cfgOut       = cfg;
        cfgOut.model = name;
        cfgOut.seed  = models(mi).seed;
        if ~ischar(models(mi).pdp); cfgOut.pdp = models(mi).pdp; end
        save(fullfile(outDir, ['H_' name '.mat']), 'Hr', 'Hi', 'cfgOut', '-v7.3');

        figure('Name', name);
        subplot(1,2,1); imagesc(Hr(:,:,1)); axis xy; colorbar;
        title([name ' Real(H)']); xlabel('OFDM symbol'); ylabel('Subcarrier');
        subplot(1,2,2); imagesc(Hi(:,:,1)); axis xy; colorbar;
        title([name ' Imag(H)']); xlabel('OFDM symbol'); ylabel('Subcarrier');
    end
end
