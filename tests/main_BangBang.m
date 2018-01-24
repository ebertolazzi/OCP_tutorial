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

addpath('../lib') ;

N     = 1000 ;
nodes = (0:N)/N ;

bb = OCP_BangBang() ;

bb.setup( nodes ) ;

info = bb.solve() ;

bb.plot() ;

% x  = sol(1:2:2*N) ;
% v  = sol(2:2:2*N) ;
% ua = sol(2*N+(1:2:2*N-2)) ;
% ub = sol(2*N+(2:2:2*N-2)) ;
% 
% subplot( 3, 1, 1 );  
% plot( nodes, x ) ;
% 
% subplot( 3, 1, 2 );  
% plot( nodes, v ) ;
% 
% subplot( 3, 1, 3 );  
% plot( nodes(1:end-1), ua, nodes(1:end-1), ub ) ;
% 
info 