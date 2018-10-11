%This file is part of ME4PyUtils, This module aims to ease some data type 
%conversion between matlab engine API for python and numpy python module.
%Copyright (C) 2018 -- Benoit Nennig, benoit.nennig@supmeca.fr

%This program is free software: you can redistribute it and/or modify
%it under the terms of the GNU General Public License as published by
%the Free Software Foundation, either version 3 of the License, or
%(at your option) any later version.

%This program is distributed in the hope that it will be useful,
%but WITHOUT ANY WARRANTY; without even the implied warranty of
%MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
%GNU General Public License for more details.

%You should have received a copy of the GNU General Public License
%along with this program.  If not, see <http://www.gnu.org/licenses/>.

function varargout= Recast4py(varargin)

% This function aims to make the interface with an existing matlab function 
% and python (matlab python engine) easier when the function return sparse matrices.
% All the input are mapped to the output except if sparse matrices are present.
% In this case the matrix is put COOrdinate format in a python dict : 
% ['i'] -> row index
% ['j'] -> column index
% ['c'] -> value

% example : if [A,B,C]=Myfunc(a,b,c) existe, create 
%	    function [A,B,C]=Myfunc_4py(a,b,c)
%           [A,B,C]=Myfunc(a,b,c)
%           [A,B,C]=Recast4py(A,B,C)


if nargin~=nargout
    error('The number of input must be equal to the number of ouput')
end
% loop over the input
for k = 1:nargin
    
    % if input is a sparse martix
    if  issparse(varargin{k})
        [ii,jj,ss] = find(varargin{k});
        % conversion to structure (dict in python)
        shape = size(varargin{k})
        temp = struct('i',int64(ii),'j',int64(jj),'s',ss,'shape',int64(shape));
        varargout{k}=temp;
    else
        % do nothing...
        varargout{k} = varargin{k};
    end
    
end


