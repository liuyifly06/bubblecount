function DrawCircle(x, y, r, nseg, S)
% Draw a circle on the current figure using ploylines
%
%  DrawCircle(x, y, r, nseg, S)
%  A simple function for drawing a circle on graph.
%
%  INPUT: (x, y, r, nseg, S)
%  x, y:    Center of the circle
%  r:       Radius of the circle
%  nseg:    Number of segments for the circle
%  S:       Colors, plot symbols and line types
%
%  OUTPUT: None

theta = 0 : (2 * pi / nseg) : (2 * pi);
pline_x = r * cos(theta) + x;
pline_y = r * sin(theta) + y;

plot(pline_x, pline_y, S);
