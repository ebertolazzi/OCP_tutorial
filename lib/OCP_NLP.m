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

classdef (Abstract) OCP_NLP < handle

  properties (SetAccess = protected, Hidden = true)
    nodes
    N      % total number of nodes
    nx     % number of states
    nu     % number of controls
    np     % number of parameters
    nbc    % number of boundary conditions
  end

  methods (Abstract)
    % Lagrange target
    L     = lagrange( self, tL, tR, XL, XR, UC, PARS )
    gradL = lagrange_gradient( self, tL, tR, XL, XR, UC, PARS )
    hessL = lagrange_hessian( self, tL, tR, XL, XR, UC, PARS )

    % Mayer target
    M     = mayer( self, tL, tR, XL, XR, PARS )
    gradM = mayer_gradient( self, tL, tR, XL, XR, PARS )
    hessM = mayer_hessian( self, tL, tR, XL, XR, PARS )

    % Dynamical system part
    C   = ds( self, tL, tR, XL, XR, UC, PARS )
    CJ  = ds_jacobian( self, tL, tR, XL, XR, UC, PARS )
    CH  = ds_hessian( self, tL, tR, XL, XR, UC, PARS, L )

    % Jump condition
    jmp  = jump( self, tL, tR, XL, XR, UC, PARS )
    jmpJ = jump_jacobian( self, tL, tR, XL, XR, UC, PARS )
    jmpH = jump_hessian( self, tL, tR, XL, XR, UC, PARS, L )

    % Boundary conditions
    bcf = bc( self, tL, tR, XL, XR, PARS )
    bcJ = bc_jacobian( self, tL, tR, XL, XR, PARS )
    bcH = bc_hessian( self, tL, tR, XL, XR, PARS, L )
  end

  methods

    function self = OCP_NLP( nx, nu, np, nbc )
      self.nx  = nx ;
      self.nu  = nu ;
      self.np  = np ;
      self.nbc = nbc ;
    end

    function setup( self, nodes )
      self.nodes = nodes;
      self.N     = length(nodes) ;
    end
    
    %  _                     _
    % | |_ __ _ _ _ __ _ ___| |_
    % |  _/ _` | '_/ _` / -_)  _|
    %  \__\__,_|_| \__, \___|\__|
    %              |___/
    %
    function res = NLP_target( self, Z )

      totx = self.N*self.nx ;
      totu = (self.N-1)*self.nu ;

      idx  = 1:self.nx ;
      idx1 = (totx-self.nx)+(1:self.nx) ;
      idp  = (totx+totu)+(1:self.np) ;

      res = self.mayer( self.nodes(1), self.nodes(end), Z(idx), Z(idx1), Z(idp) ) ;
      idu = totx+(1:self.nu) ;
      for k=1:self.N-1
        nk   = self.nodes(k) ;
        nk1  = self.nodes(k+1) ;
        idx1 = idx + self.nx ; 
        res  = res + self.lagrange( nk, nk1, Z(idx), Z(idx1), Z(idu), Z(idp) ) ;
        idu  = idu + self.nu ;
        idx  = idx1 ;
      end
      % variation for controls
      % for k=2:N-1
      %   res = res + (nodes(k+1)-nodes(k-1)) * data.pars.uepsi * sum( (U{k}-U{k-1}).^2 ) ;
      % end
    end

    %
    function g = NLP_target_gradient( self, Z )

      totx = self.N*self.nx ;
      totu = (self.N-1)*self.nu ;
      
      idx  = 1:self.nx ;
      idx1 = (totx-self.nx)+(1:self.nx) ;
      idp  = (totx+totu)+(1:self.np) ;
      
      g = zeros( 1, totx + totu + self.np ) ;
      g([idx,idx1,idp]) = self.mayer_gradient( self.nodes(1), self.nodes(end), Z(idx), Z(idx1), Z(idp) ) ;
      idu = totx+(1:self.nu) ;
      for k=1:self.N-1
        nk    = self.nodes(k) ;
        nk1   = self.nodes(k+1) ;
        idx1  = idx + self.nx ;
        id    = [ idx, idx1, idu, idp ] ;
        g(id) = g(id) + self.lagrange_gradient( nk, nk1, Z(idx), Z(idx1), Z(idu), Z(idp) ) ;
        idx   = idx1 ;
        idu   = idu + self.nu ;
      end
      % variation for controls
      % for k=2:N-1
      %   tmp     = 2 * (nodes(k+1)-nodes(k-1)) * data.pars.uepsi * (U{k}-U{k-1}).' ;
      %   gu{k}   = gu{k}   + tmp ; 
      %   gu{k-1} = gu{k-1} - tmp ;
      % end
      %g = [ cell2mat(gx), cell2mat(gu), gp ] ;
    end

    %  _  _           _
    % | || |___ _____(_)__ _ _ _
    % | __ / -_|_-<_-< / _` | ' \
    % |_||_\___/__/__/_\__,_|_||_|
    %
    function H = NLP_hessian( self, Z, sigma, lambda )

      totx = self.N*self.nx ;
      totu = (self.N-1)*self.nu ;
      H    = sparse( totx+totu, totx+totu ) ;

      idx  = 1:self.nx ;
      idx1 = (totx-self.nx)+(1:self.nx) ;
      idp  = (totx+totu)+(1:self.np) ;
      idl  = (self.nx*(self.N-1))+(1:self.nbc);

      imap = [ idx, idx1, idp ] ;
      n1   = self.nodes(1) ;
      ne   = self.nodes(end) ;
      H(imap,imap) = sigma * self.mayer_hessian( n1, ne, Z(idx), Z(idx1), Z(idp) ) + ...
                     self.bc_hessian( n1, ne, Z(idx), Z(idx1), Z(idp), lambda(idl) ) ;
      idu = totx+(1:self.nu) ;
      for k=1:self.N-1
        nk   = self.nodes(k) ;
        nk1  = self.nodes(k+1) ;
        idx1 = idx + self.nx ;
        imap = [ idx, idx1, idu, idp ] ;
        H(imap,imap) = H(imap,imap) + ...
                       sigma * self.lagrange_hessian( nk, nk1, Z(idx), Z(idx1), Z(idu), Z(idp) ) + ...
                       self.ds_hessian( nk, nk1, Z(idx), Z(idx1), Z(idu), Z(idp), lambda(idx) ) ;
        idx = idx1 ;
        idu = idu + self.nu ;
      end
      %% variation for controls
      %for k=2:N-1
      %  imap = [ totx+(k-2)*nu+(1:2*nu) ] ;
      %  tmp  = 2 * sigma * (nodes(k+1)-nodes(k-1)) * data.pars.uepsi ;
      %  H(imap,imap) = H(imap,imap) + tmp * [ eye(nu), -eye(nu) ; -eye(nu), eye(nu) ] ;
      %end
      H = tril(H) ;
    end

    function H = NLP_hessian_pattern( self )
      totx = self.N*self.nx ;
      totu = (self.N-1)*self.nu ;
      H    = sparse( totx + totu, totx + totu ) ;
      imap = [1:self.nx, (totx-self.nx)+(1:self.nx) ] ;
      H(imap,imap) = ones(2*self.nx,2*self.nx);

      idp = (totx+totu)+(1:self.np) ;
      dim = 2*self.nx+self.nu+self.np ;
      for k=1:self.N-1
        idx  = (k-1)*self.nx ;
        idu  = (k-1)*self.nu + totx ;
        imap = [ idx+(1:2*self.nx), idu+(1:self.nu), idp ] ;
        H(imap,imap) = ones( dim, dim ) ;
      end
      % variation for controls
      %for k=2:self.N-1
      %  imap = [ totx+(k-2)*self.nu+(1:2*self.nu) ] ;
      %  H(imap,imap) = H(imap,imap) + [ eye(self.nu), eye(self.nu) ; eye(self.nu), eye(self.nu) ] ;
      %end
      H = tril(H) ;
    end

    %   ___             _            _     _
    %  / __|___ _ _  __| |_ _ _ __ _(_)_ _| |_ ___
    % | (__/ _ \ ' \(_-<  _| '_/ _` | | ' \  _(_-<
    %  \___\___/_||_/__/\__|_| \__,_|_|_||_\__/__/
    %
    function C = NLP_constraints( self, Z )

      totx = self.N*self.nx ;
      totu = (self.N-1)*self.nu ;
      idx  = 1:self.nx ;
      idu  = totx+(1:self.nu) ;
      idp  = (totx+totu)+(1:self.np) ;

      C = zeros( (self.N-1)*self.nx + self.nbc, 1 ) ;

      for k=1:self.N-1
        nk     = self.nodes(k) ;
        nk1    = self.nodes(k+1) ;
        idx1   = idx + self.nx ;
        C(idx) = self.ds( nk, nk1, Z(idx), Z(idx1), Z(idu), Z(idp) ) ;
        idx    = idx1 ;
        idu    = idu + self.nu ;
      end
      idx   = 1:self.nx ;
      id    = ((self.N-1)*self.nx)+(1:self.nbc) ;
      C(id) = self.bc( self.nodes(1), self.nodes(end), Z(idx), Z(idx1), Z(idp) ) ;
    end

    % -------
    function Jac = NLP_constraints_jacobian( self, Z )

      totx = self.N*self.nx ;
      totu = (self.N-1)*self.nu ;
      idx  = 1:self.nx ;
      idu  = totx+(1:self.nu) ;
      idp  = (totx+totu)+(1:self.np) ;

      dim  = (self.N-1)*self.nx+self.nbc ;
      Jac  = sparse( dim, totx+totu+self.np ) ;

      for k=1:self.N-1
        nk   = self.nodes(k) ;
        nk1  = self.nodes(k+1) ;
        idx1 = idx + self.nx ;
        J    = self.ds_jacobian( nk, nk1, Z(idx), Z(idx1), Z(idu), Z(idp) ) ;
        Jac( idx, [ idx, idx1, idu, idp ] ) = J ;
        idu = idu + self.nu ;
        idx = idx1 ;
      end
      idx = 1:self.nx ;
      J = self.bc_jacobian( self.nodes(1), self.nodes(end), Z(idx), Z(idx1), Z(idp) ) ;
      imap = [ totx - self.nx + (1:self.nbc) ] ;
      jmap = [ idx, idx1, idp ] ;
      Jac(imap,jmap) = Jac(imap,jmap) + J ;
    end

    % -------
    function Jac = NLP_constraints_jacobian_pattern( self )
      totx = self.N*self.nx ;
      totu = (self.N-1)*self.nu ;
      dimC = (self.N-1)*self.nx+self.nbc ;
      dimZ = totx+totu+self.np ;
      Jac  = sparse( dimC, dimZ ) ;
      J    = ones(self.nx,2*self.nx+self.nu+self.np) ;
      idp  = totx + totu + (1:self.np) ;
      for k=1:self.N-1
        idx  = (k-1)*self.nx ;
        idu  = (k-1)*self.nu + totx ;
        imap = idx + (1:self.nx) ;
        jmap = [ idx + (1:2*self.nx), idu + (1:self.nu), idp  ] ;
        Jac(imap,jmap) = J ;
      end
      imap = totx - self.nx + (1:self.nbc) ;
      jmap = [ 1:self.nx, totx-self.nx+(1:self.nx), idp ] ;
      Jac(imap,jmap) = ones(self.nbc,2*self.nx+self.np)  ;
    end

    %  _   _ _   _ _
    % | | | | |_(_) |___
    % | |_| |  _| | (_-<
    %  \___/ \__|_|_/__/
    %
    function H = FD_ds_hessian( self, tL, tR, XL, XR, UC, PARS, L )
      N = 2*self.nx+self.nu+self.np ;

      id1 = 1:self.nx ;
      id2 = self.nx+id1 ;
      id3 = 2*self.nx+(1:self.nu) ;
      id4 = 2*self.nx+self.nu+(1:self.np) ;

      GRAD = @(W) self.ds_jacobian( tL, tR, XL+W(id1), XR+W(id2), UC+W(id3), PARS+W(id4)).' * L ;

      % finite difference approximation of the hessian
      % Baseed on a code by Brendan C. Wood
      % Copyright (c) 2011, Brendan C. Wood <b.wood@unb.ca>
      H = zeros(N,N);
      h = max(1,abs([XL;XR;UC;PARS]))*eps^(1/3); % ricetta di cucina
      for i=1:N
        % derivative at first point (left)
        x1    = zeros(N,1);
        x1(i) = - h(i) ;
        df1   = GRAD(x1);
        
        % derivative at second point (right)
        x2    = zeros(N,1);
        x2(i) = h(i);
        df2   = GRAD(x2);
        
        % differentiate between the two derivatives
        d2f = (df2-df1) ./ (2*h(i));
        
        % assign as column i of Hessian
        H(:,i) = d2f;
      end
      H = 0.5*(H+H.');      
    end

    %     _
    %  _ | |_  _ _ __  _ __
    % | || | || | '  \| '_ \
    %  \__/ \_,_|_|_|_| .__/
    %                 |_|
    %
    function ODE = jump_standard( ~, tL, tR, XL, XR, UC )
      ODE = XR - XL ;
    end

    %
    function JAC = jump_standard_jacobian( ~, tL, tR, XL, XR, UC )
      JAC = [ -eye(self.nx,self.nx), ...
               eye(self.nx,self.nx), ...
               zeros(self.nx, self.nu+self.np) ] ;
    end

    %
    function H = jump_standard_hessian( ~, tL, tR, XL, XR, UC, L )
      dim = 2*self.nx+self.nu+self.np ;
      H   = zeros(dim,dim) ;
    end

    %        _    _           _     _
    %  _ __ (_)__| |_ __  ___(_)_ _| |_
    % | '  \| / _` | '_ \/ _ \ | ' \  _|
    % |_|_|_|_\__,_| .__/\___/_|_||_\__|
    %              |_|
    %     
    function C = midpoint_ds( self, tL, tR, XL, XR, UC, PARS, RHS )
      tM = (tR+tL)/2 ;
      XM = (XR+XL)./2 ;
      C  = (XR-XL)/(tR-tL) - feval( RHS, tM, XM, UC, PARS ) ;
    end
 
    function CJ = midpoint_ds_jacobian( self, tL, tR, XL, XR, UC, PARS, JAC )
      tM = (tR+tL)/2 ;
      XM = (XR+XL)./2 ;
      JJ = feval( JAC, tM, XM, UC, PARS ) ;
      B1 = (-0.5)*JJ(1:self.nx,1:self.nx) ;
      B2 = JJ(1:self.nx,self.nx+1:end) ;
      bf = 1/(tR - tL) ;
      CJ = [ B1-bf*eye(self.nx), B1+bf*eye(self.nx), -B2 ] ;
    end

    function CH = midpoint_ds_hessian( self, tL, tR, XL, XR, UC, PARS, L, HESS )
      tM = (tR+tL)/2 ;
      XM = (XR+XL)./2 ;
      HH = feval( HESS, tM, XM, UC, PARS, L ) ;
      D1 = (-0.25)*HH(1:self.nx,1:self.nx) ;
      R1 = (-0.5)*HH(1:self.nx,self.nx+1:end) ;
      D2 = -HH(self.nx+1:end,self.nx+1:end) ;
      CH = [ D1,   D1,   R1 ; ...
             D1,   D1,   R1 ; ...
             R1.', R1.', D2 ] ;
    end
  end
end
