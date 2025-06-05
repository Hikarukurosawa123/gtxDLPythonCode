

###PINN ideation 

#approximate the fluorescence as a point source 

#define mueff

#resume later 

#nrel for water 
nrel = 1.33
mua_xB = mua_x(1); mua_mB = mua_m(1);
% - Tumor (if length>2, otherwise = Background)
mua_xT = mua_x(length(mua_x)); mua_mT = mua_m(length(mua_m));

% Total absorption coeffficient: Background
mua_xB = mua_xB + muaf_x/TBR; mua_mB = mua_mB + muaf_m/TBR;

muafDelta_x = muaf_x*(1-1/TBR); muafDelta_m = muaf_m*(1-1/TBR);

F_posSrc = gtxTSNumerical_MATLAB(mua_xT,mus_x,mua_mT,mus_m,muafDelta_x,muafDelta_m,eta,nrel,fx,numericalVars);

def gtxTSNumerical_MATLAB(mua_x,mus_x,mua_m,mus_m,muaf_x,muaf_m,eta,nrel,fx,numericalVars)
%{
Analytical spatially-resolved reflectance model for arbitrary fluorescence inclusion
Function to be called using parameters defined below
Michael Daly & Jacqueline Fleisig
GTx Program, UHN-TECHNA
May 2014

Inputs:
1. mua_x (absorption coeff at excitation wavelength)
2. mus_x (scattering coeff at excitation wavelength)
3. mua_m (absorption coeff at emission wavelength)
4. mus_m (scattering coeff at emission wavelength)
5. muaf_x (absorption of fluorescence at excitation wavelength)
6. muaf_m (absorption of fluorescence at excitation wavelength)
7. eta (fluorescence quantum efficiency)
8. nrel (refractive index)
9. fx (spatial frequency)
10. numericalVars (1xN) which contains
- xxS (nxm double) of x-vals of source points
- yyS (nxm double) of y-vals of source points
- zzS (nxm double) of z-vals of source points
- xxD (nxm double) of x-vals of detector points
- yyD (nxm double) of x-vals of detector points
- zzD (nxm double) of z-vals of detector points
- xFl (nx1 double) of x-vals of fluorescent points
- yFl (nx1 double) of y-vals of fluorescent points
- zSlices (1xn double) of fluorescence slice depth (note: this n
    must be consistent with the n of the cell array for vars 1 and 2)
- IsFl (nxm double) of whether a point fluoresces (1) or not (0) at a given depth
- rDelta (pixel size)
- zDelta (slice thickness)
- z0_mApply (boundary conditions)

Outputs:
1. F (nxm double) fluorescence image
2. R (nxm double) reflectance image
%}

% Numerical input structure
varsToAdd = {'xxS','yyS','zzS','xxD','yyD','zzD','xFl','yFl','zSlices','IsFl','rDelta','zDelta','z0_mApply','r1min','units','methodConv','fxDependence','TBR','muaTotal'};
for vv = 1:length(varsToAdd)
    eval([varsToAdd{vv},' = numericalVars.',varsToAdd{vv},';']);
end

% Total absorption coeffficient
if strcmp(muaTotal,'C+F')
    mua_x = mua_x + muaf_x;
    mua_m = mua_m + muaf_m;
elseif strcmp(muaTotal,'COnly')
    % Just use chromophore mua_x, mua_m
end

% Fluorescence Yield ("Quantitative Fluorescence"):
qF = eta*muaf_x;

% Fluorescence Yield Volume:
Q = qF*double(IsFl);

% Output fluorescence image
F = zeros(size(xxD));

% Radial Distances
if (methodConv == 2)
    rrSFl = sqrt(xxS.^2 + yyS.^2);
    rrDFl = sqrt(xxD.^2 + yyD.^2);
end

% Fluorescence from each slice
F_z = zeros(size(xxD,1),size(xxD,2),length(zSlices));
for iSlice = 1:length(zSlices)
    % zSlices is depth from source/detector
    deltazS = zSlices(iSlice);
    deltazD = zSlices(iSlice);
    
    if (methodConv == 1)
        % Loop point by point (SLOW)
        for iPoint = 1:length(xFl)
            if IsFl(iPoint,iSlice)
                % 2D Source/Detectors
                xxDeltaSFl = xxS - xFl(iPoint);
                yyDeltaSFl = yyS - yFl(iPoint);
                rrSFl = sqrt(xxDeltaSFl.^2 + yyDeltaSFl.^2);
                xxDeltaDFl = xxD - xFl(iPoint);
                yyDeltaDFl = yyD - yFl(iPoint);
                rrDFl = sqrt(xxDeltaDFl.^2 + yyDeltaDFl.^2);
                
                % Effective Optical Transport [1/mm]
                H = gtxTSEffectiveOptTransport(mua_x,mus_x,mua_m,mus_m,nrel,fx,rrSFl,rrDFl,rDelta,zDelta,deltazS,deltazD,z0_mApply,r1min,fxDependence);
                
                % Fluorescence [1/mm2]
                F = F + H*Q(iPoint,iSlice);
                F_z(:,:,iSlice) = F_z(:,:,iSlice) + H*Q(iPoint,iSlice);
            end
        end
    elseif (methodConv == 2)
        % Effective Optical Transport [1/mm]
        H = gtxTSEffectiveOptTransport(mua_x,mus_x,mua_m,mus_m,nrel,fx,rrSFl,rrDFl,rDelta,zDelta,deltazS,deltazD,z0_mApply,r1min,fxDependence);
        
        % Fluorescence [1/mm2]
        % - Convolve for current slice
        Q_z = reshape(Q(:,iSlice),size(xxD));
        F_z(:,:,iSlice) = conv2(Q_z,H,'same');
        % - Accumulate over slices
        F = F + F_z(:,:,iSlice);
        
        %{
        % Untested code for rDelta ~= pixelSize
        Nq = sqrt(length(xFl));
        Q_z = reshape(Q(:,iSlice),Nq,Nq);
        F_z(:,:,iSlice) = conv2(H,Q_z,'same');
        %}
    end
end

if strcmp(units,'total')
    % Multiply by detector pixel area: F [no units] = F [1/mm2] * Area [mm^2]
    F = F .* rDelta^2;
end

%{
for iSlice = 1:length(zSlices)
    figure; gtxImageDisplay(F_z(:,:,iSlice))
    title(num2str(iSlice));
end
%}

return F 

def gtxTSEffectiveOptTransport(mua_x,mus_x,mua_m,mus_m,nrel,fx,rrSFl,rrDFl,rDelta,zDelta,deltazS,deltazD,z0_mApply,r1min,fxDependence):
% Compute effective optical transport [1/mm]

% Excitation Green's functions from source positions [1/mm2]
G_x = gtxDTFluenceSpatial(mua_x,mus_x,nrel,rrSFl,deltazS,'surface',fx,'semi',r1min);

% Integrate over all sources (multiply by source area)
Phi_x = sum(G_x(:))*rDelta^2;

% Emission Green's function [1/mm2]
if ~fxDependence
    fx = 0*fx; % Dependence on fx in emission
end
if z0_mApply
    G_m = gtxDTFluenceSpatial(mua_m,mus_m,nrel,rrDFl,deltazD,'surface',fx,'semi',r1min);
else
    G_m = gtxDTFluenceSpatial(mua_m,mus_m,nrel,rrDFl,deltazD,'buried',fx,'semi',r1min);
end

% Flux boundary conditions:
Cnd = gtxDTBoundaryCondition(nrel);

% Effective Optical Transport [1/mm]
H = G_m*Phi_x*zDelta/Cnd;

return H


function Phi = gtxDTFluenceSpatial(mua,mus,nrel,r,z,source,fx,geometry,r1min)
% Compute Green's function for diffuse tissue transport
% Point-source input to homogeneous medium
% Michael Daly
% GTx-Program (UHN-TECHNA)
% July 2015

if (nargin < 7) 
    fx = 0;
end
if (nargin < 8)
    geometry = 'semi';
end
if (nargin < 9)
    r1min = -inf;
end

% Flux boundary conditions:
[~,K] = gtxDTBoundaryCondition(nrel);

mutr = mua + mus;
D = 1./(3*mutr);
mueff = sqrt(mua./D);
mueff = sqrt(mueff.*mueff + (2*pi*fx).^2);
if strcmp(source,'surface')
    z0 = 1./mutr;
elseif strcmp(source,'buried')
    z0 = 0;
else
    error('Incorrect source input');
end

% Fluence Rate (Jacques, Diffusion Review, JBO 2008)
% Phi [1/mm^2]
r1 = sqrt((z-z0).^2 + r.^2);
r1 = max(r1,r1min);
if strcmp(geometry,'inf')
    % Farrel et. al. (Med Phys 1992) [Eq. 10]
    Phi = 1/(4*pi*D)*(exp(-mueff*r1)./r1);
elseif strcmp(geometry,'semi')
    zb = 2*K*D;
    r2 = sqrt((z0+z+2*zb).^2 + r.^2);
    % Farrel et. al. (Med Phys 1992) [Eq. 12]
    Phi = 1./(4*pi*D).*(exp(-mueff*r1)./r1 - exp(-mueff*r2)./r2);
end