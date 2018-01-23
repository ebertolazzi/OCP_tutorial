%--------------------------------------------------------------------------%
%                                                                          %
%  Copyright (C) 2018                                                      %
%                                                                          %
%         , __                 , __                                        %
%        /|/  \               /|/  \                                       %
%         | __/ _   ,_         | __/ _   ,_                                %
%         |   \|/  /  |  |   | |   \|/  /  |  |   |                        %
%         |(__/|__/   |_/ \_/|/|(__/|__/   |_/ \_/|/                       %
%                           /|                   /|                        %
%                           \|                   \|                        %
%                                                                          %
%      Enrico Bertolazzi                                                   %
%      Dipartimento di Ingegneria Industriale                              %
%      Universita` degli Studi di Trento                                   %
%      email: enrico.bertolazzi@unitn.it                                   %
%                                                                          %
%--------------------------------------------------------------------------%

classdef OCP_BangBang < OCP_NLP

  properties (SetAccess = private, Hidden = true)
    x_i
    v_i
    x_f
    v_f
    sol
  end

  methods

    function self = OCP_BangBang( )
      self@OCP_NLP( 2, 1, 1, 4 ) ;
    end
    
    function setup( self, nodes )
      setup@OCP_NLP( self, nodes ) ;
      self.x_i = 0;
      self.x_f = 1;
      self.v_i = 0;
      self.v_f = 0;
    end 

    function info = solve( self )
      
      xones = ones(1,self.N*self.nx) ;
      uones = ones(1,(self.N-1)*self.nu) ;
      
      options.lb = [ -xones*Inf, -uones, 0   ] ;  % Lower bound on the variables.
      options.ub = [  xones*Inf,  uones, Inf ] ;  % Upper bound on the variables.

      % The constraint functions are bounded to zero
      options.cl = zeros(1,(self.N-1)*self.nx+self.nbc); %  constraints
      options.cu = zeros(1,(self.N-1)*self.nx+self.nbc);

      % Set the IPOPT options.
      options.ipopt.jac_d_constant   = 'no';
      options.ipopt.hessian_constant = 'no';
      options.ipopt.mu_strategy      = 'adaptive';
      options.ipopt.max_iter         = 400;
      options.ipopt.tol              = 1e-10;%
      options.ipopt.linear_solver    = 'ma57';
      %options.ipopt.linear_solver    = 'mumps';
      %options.ipopt.linear_solver    = 'pardiso';

      % The callback functions.
      funcs.objective         = @(Z) self.NLP_target(Z);
      funcs.gradient          = @(Z) self.NLP_target_gradient(Z);

      funcs.constraints       = @(Z) self.NLP_constraints(Z);
      funcs.jacobian          = @(Z) self.NLP_constraints_jacobian(Z);
      funcs.jacobianstructure = @() self.NLP_constraints_jacobian_pattern();

      if false
        %options.ipopt.derivative_test = 'second-order';
        funcs.hessian           = @( Z, sigma, lambda ) self.NLP_hessian( Z, sigma, lambda ) ;
        funcs.hessianstructure  = @()  self.NLP_hessian_pattern();
      else
        %options.ipopt.derivative_test            = 'first-order';
        options.ipopt.hessian_approximation      = 'limited-memory';
        options.ipopt.limited_memory_update_type = 'bfgs' ; % {bfgs}, sr1 = 6; % {6}
      end

      % Run IPOPT.
      xguess = (self.x_i+(self.x_f-self.x_i)*self.nodes).' ;
      vguess = zeros(self.N,1);
      uguess = zeros(self.N-1,1);

      x0 = [ reshape( [ xguess, vguess], 2*self.N ,1 ) ; uguess ; 0 ] ;

      tic
      [self.sol, info] = ipopt(x0,funcs,options);
      elapsed = toc ;

    end
    
    function plot( self )
      N     = self.N;
      x     = self.sol(1:2:2*N);
      v     = self.sol(2:2:2*N);
      u     = self.sol(2*N+1:end-1);
      T     = self.sol(end) ;
      nodes = self.nodes;

      subplot( 3, 1, 1 );  
      plot( nodes, x );

      subplot( 3, 1, 2 );  
      plot( nodes, v );

      subplot( 3, 1, 3 );  
      plot( nodes(1:end-1), u );
    end

    %                      __              _   _
    %  _  _ ___ ___ _ _   / _|_  _ _ _  __| |_(_)___ _ _  ___
    % | || (_-</ -_) '_| |  _| || | ' \/ _|  _| / _ \ ' \(_-<
    %  \_,_/__/\___|_|   |_|  \_,_|_||_\__|\__|_\___/_||_/__/
    %

    %  _
    % | |   __ _ __ _ _ _ __ _ _ _  __ _ ___
    % | |__/ _` / _` | '_/ _` | ' \/ _` / -_)
    % |____\__,_\__, |_| \__,_|_||_\__, \___|
    %           |___/              |___/
    %
    function L = lagrange( ~, tL, tR, XL, XR, UC, P )
      L = 0 ;
    end

    %
    function gradL = lagrange_gradient( self, tL, tR, XL, XR, UC, P )
      dim   = 2*self.nx + self.nu + self.np ;
      gradL = zeros( 1, dim ) ;
    end

    %
    function hessL = lagrange_hessian( self, tL, tR, XL, XR, UC, P )
      dim   = 2*self.nx + self.nu + self.np ;
      hessL = zeros(dim,dim) ;
    end

    %  __  __
    % |  \/  |__ _ _  _ ___ _ _
    % | |\/| / _` | || / -_) '_|
    % |_|  |_\__,_|\_, \___|_|
    %              |__/
    %
    function M = mayer( ~, tL, tR, XL, XR, T )
      M = T ; % only one parameter
    end
    
    %
    function gradM = mayer_gradient( self, tL, tR, XL, XR, T )
      dim   = 2*self.nx + self.np ;
      gradM = zeros(1,dim) ;
      gradM(2*self.nx+1) = 1 ;
    end
    
    % [ M, gradM, hessianM ]
    function hessM = mayer_hessian( self, tL, tR, XL, XR, T )
      dim   = 2*self.nx + self.np ;
      hessM = zeros(dim,dim) ;
    end

    %   ___  ___  ___   _____   _   ___
    %  / _ \|   \| __| / /   \ /_\ | __|
    % | (_) | |) | _| / /| |) / _ \| _|
    %  \___/|___/|___/_/ |___/_/ \_\___|
    %
    function C = ds( self, tL, tR, XL, XR, UC, T )
      xL = XL(1) ; vL = XL(2) ;
      xR = XR(1) ; vR = XR(2) ;
      u  = UC(1) ;
      % ----------
      DT = tR - tL ;
      xM = (xR+xL)/2 ;
      vM = (vR+vL)/2 ;
      % ----------
      C    = zeros(2,1) ;
      C(1) = (xR - xL)/DT - T*vM ;
      C(2) = (vR - vL)/DT - T*u ;
    end

    %
    function JAC = ds_jacobian( self, tL, tR, XL, XR, UC, T )
      xL = XL(1) ; vL = XL(2) ;
      xR = XR(1) ; vR = XR(2) ;
      u  = UC(1) ;
      % ----------
      DT = tR - tL ;
      xM = (xR+xL)/2 ;
      vM = (vR+vL)/2 ;
      % ----------
      JAC = [ -1/DT, -0.5*T,  1/DT, -0.5*T,  0, -vM ; ... 
                  0,  -1/DT,     0,   1/DT, -T, -u ] ;
    end

    %
    function H = ds_hessian( self, tL, tR, XL, XR, UC, T, L )
      xL = XL(1) ; vL = XL(2) ;
      xR = XR(1) ; vR = XR(2) ;
      u  = UC(1) ;
      % ----------
      DT = tR - tL ;
      xM = (xR+xL)/2 ;
      vM = (vR+vL)/2 ;
      % ----------
      H1 = [ 0,    0, 0,    0, 0,    0 ; ...
             0,    0, 0,    0, 0, -0.5 ; ...
             0,    0, 0,    0, 0,    0 ; ...
             0,    0, 0,    0, 0, -0.5 ; ...
             0,    0, 0,    0, 0,    0 ; ...
             0, -0.5, 0, -0.5, 0,    0 ] ;
      H2 = [ 0, 0, 0, 0,  0,  0 ; ...
             0, 0, 0, 0,  0,  0 ; ...
             0, 0, 0, 0,  0,  0 ; ...
             0, 0, 0, 0,  0,  0 ; ...
             0, 0, 0, 0,  0, -1 ; ...
             0, 0, 0, 0, -1,  0 ] ;
      H = L(1) * H1 + L(2) * H2 ;
    end

    %     _
    %  _ | |_  _ _ __  _ __
    % | || | || | '  \| '_ \
    %  \__/ \_,_|_|_|_| .__/
    %                 |_|
    %
    function ODE = jump( ~, tL, tR, XL, XR, UC )
      ODE = XR - XL ;
    end

    %
    function JAC = jump_jacobian( ~, tL, tR, XL, XR, UC )
      JAC = [ -eye(self.nx,self.nx), ...
               eye(self.nx,self.nx), ...
               zeros(self.nx, self.nu+self.np) ] ;
    end

    %
    function H = jump_hessian( ~, tL, tR, XL, XR, UC, L )
      dim = 2*self.nx+self.nu+self.np ;
      H   = zeros(dim,dim) ;
    end

    %  ___  ___
    % | _ )/ __|
    % | _ \ (__
    % |___/\___|
    %
    function bc = bc( self, tL, tR, XL, XR, P )
      xL = XL(1) ; vL = XL(2) ;
      xR = XR(1) ; vR = XR(2) ;
      bc = [ xL - self.x_i ; ...
             xR - self.x_f ; ...
             vL - self.v_i ; ...
             vR - self.v_f ] ;
    end

    %
    function Jac = bc_jacobian( ~, tL, tR, XL, XR, P )
      Jac = [ 1, 0, 0, 0, 0 ; ...
              0, 0, 1, 0, 0 ; ...
              0, 1, 0, 0, 0 ; ...
              0, 0, 0, 1, 0 ] ;
    end
    
    %
    function Hess = bc_hessian( ~, tL, tR, XL, XR, P, L )
      Hess = zeros(5,5) ;
    end

  end
end
