function show_psolution(nodes, cells, solution, fname)
	figure; title('Approximate Solution');
	MNV = max(cellfun(@numel, cells));
	f = @(vlist) [double(vlist) nan(1, MNV - numel(vlist))];
	cells = cellfun(f, cells, 'UniformOutput', false);
	cells = vertcat(cells{:});
	data = [nodes, solution'];
	patch('Faces', cells, 'Vertices', data,...
							'FaceColor', 'interp', 'CData', solution/max(abs(solution)));
	axis('square')
	xlim([min(nodes(:, 1)) - 0.1, max(nodes(:, 1)) + 0.1])
	ylim([min(nodes(:, 2)) - 0.1, max(nodes(:, 2)) + 0.1])
	zlim([min(solution) - 0.1, max(solution) + 0.1])
	xlabel('x'); ylabel('y'); zlabel('u');
    saveas(gcf, fname)
end
