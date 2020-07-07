
% clc; 

% set(0, 'DefaultFigureVisible', 'on');
if exist('figure/') == 0
	mkdir('figure/');
end

indExtend = 1;

char srcfilename;
char saveName;

KN = [80, 80, 80];
Nd=64;
fALft = 14;
fARgt =14;	

fBLft = 29;
fBRgt = 29;

fCLft = 11.8;
fCRgt = 11.8;



for fA = fALft:1:fARgt
	for fB = fBLft:1:fBRgt
        for fC = fCLft:1:fCRgt
		    fD = 100-fA-fB-fC;

%	pause(2)

%             char parameters;
%             parameters = sprintf('[fA, fB, fC, fD]: [%f,%f,%f,%f]', fA, fB, fC,fD);
%             fprintf('parameters: %s\n', parameters);
% 
%             fnA = fA / 100;
%             fnB = fB / 100;
%             fnC = fC / 100;
%             fnD = fD / 100;
%             srcfilename = sprintf('phiA.[%.2f.%.2f.%.2f].[%.4f.%.4f.%.4f.%.4f]-[%d*%d].dat',...
%                 KN(1), KN(2), KN(3), fnA, fnB, fnC, fnD, Nd,Nd);

% %    if (fA < 10)
% %        srcfilename = sprintf('phiA.[30.00.30.00.30.00].[0.0%d.0.%d.0.%d]-[128].dat', fA, fB, fC);
% %    elseif (fB < 10)
% %        srcfilename = sprintf('phiA.[30.00.30.00.30.00].[0.%d.0.0%d.0.%d]-[128].dat', fA, fB, fC);
% %    elseif (fC < 10)
% %        srcfilename = sprintf('phiA.[30.00.30.00.30.00].[0.%d.0.%d.0.0%d]-[128].dat', fA, fB, fC);
% %    else
% %        srcfilename = sprintf('phiA.[30.00.30.00.30.00].[0.%d.0.%d.0.%d]-[128].dat', fA, fB, fC);
% %    end
% 
%             DataA = load(strrep(srcfilename, 'A', 'A'));
%             DataB = load(strrep(srcfilename, 'A', 'B1'));
%             DataC = load(strrep(srcfilename, 'A', 'C'));
%             DataD = load(strrep(srcfilename, 'A', 'B2'));
            load('rho.mat');
%             [Line, Column] = size(DataA);
%             x = load('Lx.[80.00.80.00.80.00].[0.1400.0.2900.0.1180.0.4520]-[64*64].dat');
            x = linspace(0,4.0,64);y = linspace(0,4,64);
%             x=x(1:end-1);
%             x=x(1:end-1);
%             y = load('Ly.[80.00.80.00.80.00].[0.1400.0.2900.0.1180.0.4520]-[64*64].dat');
%             y=y(1:end-1);
            nz = 2;
            z = 1:nz; 
            xrange=max(x)-min(x);
            yrange=max(y)-min(y);
            zrange=max(z)-min(z);

            [X Y Z] = meshgrid(x, y, z);

            for i = 1:nz
                phiA(:,:,i) = rhoA;
                phiB(:,:,i) = rhoB;
                phiC(:,:,i) = rhoC;
            end

            axis off
            axis equal
            set(gcf, 'color', 'white')

minphiA = min(phiA(:));
maxphiA = max(phiA(:));
isoA = minphiA+0.7*(maxphiA-minphiA);
%        	   isoA = 0.1;
minphiB = min(phiB(:));
maxphiB = max(phiB(:));
isoB = minphiB+0.7*(maxphiB-minphiB);
%             isoB = 0.1;
minphiC = min(phiC(:));
maxphiC = max(phiC(:));
isoC = minphiC+0.7*(maxphiC-minphiC);
%             isoC = 0.9;

%             fprintf('Line=%d, Column=%d;  isoA=%.4f, isoB=%.4f, isoC=%.4f\n',...
%                 Line, Column, isoA, isoB, isoC);

            xextend = X+xrange;
            yextend = Y+yrange;
            zextend = Z+zrange;

            alpA = 1.0;
            alpB = 1.0;
            alpC = 1.0;

            figure(1)
            if indExtend == 0

                set(gcf, 'color', 'white')
                title('phi');
                axis off
                axis equal
                patch(isosurface(X,Y,Z, phiA, isoA), ...
                'facecolor','green', 'FaceAlpha', alpA, 'edgecolor','none');
                patch(   isocaps(X,Y,Z, phiA, isoA, 'enclose'), ...
                'facecolor','green','FaceAlpha', alpA, 'edgecolor','none');

                patch(isosurface(X,Y,Z, phiB, isoB), ... 
                'facecolor','blue','FaceAlpha', alpB, 'edgecolor','none');
                patch(   isocaps(X,Y,Z, phiB, isoB,'enclose'), ...
                'facecolor','blue','FaceAlpha', alpB, 'edgecolor','none');

                patch(isosurface(X,Y,Z, phiC, isoC), ...
                'facecolor','red','FaceAlpha', alpC, 'edgecolor','none');
                patch(   isocaps(X,Y,Z, phiC, isoC,'enclose'), ...
                'facecolor','red','FaceAlpha', alpC, 'edgecolor','none');
                colorbar;

                patch(isosurface(X,Y,Z, phiD, isoB), ...
                'facecolor','blue','FaceAlpha', alpB, 'edgecolor','none');
                patch(   isocaps(X,Y,Z, phiD, isoB,'enclose'), ...
                'facecolor','blue','FaceAlpha', alpB, 'edgecolor','none');
                colorbar;
            else
                patch(isosurface(X,Y,Z, phiA, isoA), ...
                'facecolor','green', 'FaceAlpha', alpA, 'edgecolor','none');
                patch(   isocaps(X,Y,Z, phiA, isoA, 'enclose'), ...
                'facecolor','green','FaceAlpha', alpA, 'edgecolor','none');
                patch(isosurface(xextend,Y,Z, phiA, isoA),    'FaceAlpha', alpA,        'facecolor','green','edgecolor','none');
                patch(   isocaps(xextend,Y,Z, phiA, isoA ,'enclose'),'facecolor','green','FaceAlpha', alpA, 'edgecolor','none');
                patch(isosurface(X,yextend,Z, phiA, isoA),           'facecolor','green','FaceAlpha', alpA, 'edgecolor','none');
                patch(   isocaps(X,yextend,Z, phiA, isoA ,'enclose'),'facecolor','green','FaceAlpha', alpA, 'edgecolor','none');
                patch(isosurface(X,Y,zextend, phiA, isoA),           'facecolor','green','FaceAlpha', alpA, 'edgecolor','none');
                patch(   isocaps(X,Y,zextend, phiA, isoA ,'enclose'),'facecolor','green','FaceAlpha', alpA, 'edgecolor','none');
                patch(isosurface(xextend,yextend,Z, phiA, isoA),           'facecolor','green','FaceAlpha', alpA, 'edgecolor','none');
                patch(   isocaps(xextend,yextend,Z, phiA, isoA ,'enclose'),'facecolor','green','FaceAlpha', alpA, 'edgecolor','none');
                patch(isosurface(xextend,Y,zextend, phiA, isoA),           'facecolor','green','FaceAlpha', alpA, 'edgecolor','none');
                patch(   isocaps(xextend,Y,zextend, phiA, isoA ,'enclose'),'facecolor','green','FaceAlpha', alpA, 'edgecolor','none');
                patch(isosurface(X,yextend,zextend, phiA, isoA),           'facecolor','green','FaceAlpha', alpA, 'edgecolor','none');
                patch(   isocaps(X,yextend,zextend, phiA, isoA ,'enclose'),'facecolor','green','FaceAlpha', alpA, 'edgecolor','none');
                patch(isosurface(xextend,yextend,zextend, phiA, isoA),           'facecolor','green','FaceAlpha', alpA, 'edgecolor','none');
                patch(   isocaps(xextend,yextend,zextend, phiA,  isoA,'enclose'),'facecolor','green','FaceAlpha', alpA, 'edgecolor','none');


                patch(isosurface(X,Y,Z, phiB, isoB), ... 
                'facecolor','blue','FaceAlpha', alpB, 'edgecolor','none');
                patch(   isocaps(X,Y,Z, phiB, isoB,'enclose'), ...
                'facecolor','blue','FaceAlpha', alpB, 'edgecolor','none');
                patch(isosurface(xextend,Y,Z, phiB, isoB), 'facecolor','blue','FaceAlpha', alpB, 'edgecolor','none');
                patch(   isocaps(xextend,Y,Z, phiB, isoB ,'enclose'),'facecolor','blue','FaceAlpha', alpB, 'edgecolor','none');
                patch(isosurface(X,yextend,Z, phiB, isoB), 'facecolor','blue','FaceAlpha', alpB, 'edgecolor','none');
                patch(   isocaps(X,yextend,Z, phiB, isoB ,'enclose'),'facecolor','blue','FaceAlpha', alpB, 'edgecolor','none');
                patch(isosurface(X,Y,zextend, phiB, isoB), 'facecolor','blue','FaceAlpha', alpB, 'edgecolor','none');
                patch(   isocaps(X,Y,zextend, phiB, isoB ,'enclose'),'facecolor','blue','FaceAlpha', alpB, 'edgecolor','none');
                patch(isosurface(xextend,yextend,Z, phiB, isoB), 'facecolor','blue','FaceAlpha', alpB, 'edgecolor','none');
                patch(   isocaps(xextend,yextend,Z, phiB, isoB ,'enclose'),'facecolor','blue','FaceAlpha', alpB, 'edgecolor','none');
                patch(isosurface(xextend,Y,zextend, phiB, isoB), 'facecolor','blue','FaceAlpha', alpB, 'edgecolor','none');
                patch(   isocaps(xextend,Y,zextend, phiB, isoB ,'enclose'),'facecolor','blue','FaceAlpha', alpB, 'edgecolor','none');
                patch(isosurface(X,yextend,zextend, phiB, isoB), 'facecolor','blue','FaceAlpha', alpB, 'edgecolor','none');
                patch(   isocaps(X,yextend,zextend, phiB, isoB ,'enclose'),'facecolor','blue','FaceAlpha', alpB, 'edgecolor','none');
                patch(isosurface(xextend,yextend,zextend, phiB, isoB), 'facecolor','blue','FaceAlpha', alpB, 'edgecolor','none');
                patch(   isocaps(xextend,yextend,zextend, phiB, isoB,'enclose'),'facecolor','blue','FaceAlpha', alpB, 'edgecolor','none');

                patch(isosurface(X,Y,Z, phiC, isoC), ...
                'facecolor','red','FaceAlpha', alpC, 'edgecolor','none');
                patch(   isocaps(X,Y,Z, phiC, isoC,'enclose'), ...
                'facecolor','red','FaceAlpha', alpC, 'edgecolor','none');
                patch(isosurface(xextend,Y,Z, phiC, isoC), 'facecolor','red','FaceAlpha', alpC, 'edgecolor','none');
                patch(   isocaps(xextend,Y,Z, phiC, isoC ,'enclose'),'facecolor','red','FaceAlpha', alpC, 'edgecolor','none');
                patch(isosurface(X,yextend,Z, phiC, isoC), 'facecolor','red','FaceAlpha', alpC, 'edgecolor','none');
                patch(   isocaps(X,yextend,Z, phiC, isoC ,'enclose'),'facecolor','red','FaceAlpha', alpC, 'edgecolor','none');
                patch(isosurface(X,Y,zextend, phiC, isoC), 'facecolor','red','FaceAlpha', alpC, 'edgecolor','none');
                patch(   isocaps(X,Y,zextend, phiC, isoC ,'enclose'),'facecolor','red','FaceAlpha', alpC, 'edgecolor','none');
                patch(isosurface(xextend,yextend,Z, phiC, isoC), 'facecolor','red','FaceAlpha', alpC, 'edgecolor','none');
                patch(   isocaps(xextend,yextend,Z, phiC, isoC ,'enclose'),'facecolor','red','FaceAlpha', alpC, 'edgecolor','none');
                patch(isosurface(xextend,Y,zextend, phiC, isoC), 'facecolor','red','FaceAlpha', alpC, 'edgecolor','none');
                patch(   isocaps(xextend,Y,zextend, phiC, isoC ,'enclose'),'facecolor','red','FaceAlpha', alpC, 'edgecolor','none');
                patch(isosurface(X,yextend,zextend, phiC, isoC), 'facecolor','red','FaceAlpha', alpC, 'edgecolor','none');
                patch(   isocaps(X,yextend,zextend, phiC, isoC ,'enclose'),'facecolor','red','FaceAlpha', alpC, 'edgecolor','none');
                patch(isosurface(xextend,yextend,zextend, phiC, isoC), 'facecolor','red','FaceAlpha', alpC, 'edgecolor','none');
                patch(   isocaps(xextend,yextend,zextend, phiC, isoC,'enclose'),'facecolor','red','FaceAlpha', alpC, 'edgecolor','none');
                colorbar;
            end

            view(-90, 90)
            camlight
            lighting gouraud          
% 
%             saveName = sprintf('phi.[%.2f.%.2f.%.2f].[%.4f.%.4f.%.4f.%.4f]-[%d]',...
%                 KN(1), KN(2), KN(3), fnA, fnB, fnC, fnD, Nd);
    
%	if (fA < 10)
    %	 saveName = sprintf('phi.[30.00.30.00.30.00].[0.0%d.0.%d.0.%d]-[128]', fA, fB, fC);
    %	elseif (fB < 10)
    %	saveName = sprintf('phi.[30.00.30.00.30.00].[0.%d.0.0%d.0.%d]-[128]', fA, fB, fC);
    %	elseif (fC < 10)
    %	saveName = sprintf('phi.[30.00.30.00.30.00].[0.%d.0.%d.0.0%d]-[128]', fA, fB, fC);
    %	else
    %	saveName = sprintf('phi.[30.00.30.00.30.00].[0.%d.0.%d.0.%d]-[128]', fA, fB, fC);
    %	end
    
%             title(saveName);
%             set(gcf, 'unit', 'normalized', 'position', [0.05, 0.1, 0.8, 0.8])
%             f = getframe(gcf);
%             imwrite(f.cdata,  ['figure/', saveName, '.png']);
    %	pause(5)
%             close all
%             fprintf('finish!\n\n');



    %   画出每个rho的密度分布
%             if DIM == 2
%                [Line2, Column2] = size(DataA);
% 
%                xx = DataA(1,:);
%                yy = DataA(2,:);
%                xxrange=max(xx)-min(xx);
%                yyrange=max(yy)-min(yy);
% 
%                [XX YY] = meshgrid(xx,yy);
% 
%                Line2 = Line2-2;
%                for i = 1:Line2
%                    phi2A(i,:) = DataA(i+2, :); 
%                    phi2B(i,:) = DataB(i+2, :); 
%                    phi2C(i,:) = DataC(i+2, :); 
%                    phi2D(i,:) = DataD(i+2, :); 
%                end
% 
%                minphiA = min(phi2A(:));
%                maxphiA = max(phi2A(:));
%                isoA = minphiA+0.7*(maxphiA-minphiA)
%             %    isoA =
%                minphiB = min(phi2B(:));
%                maxphiB = max(phi2B(:));
%                isoB = minphiB+0.7*(maxphiB-minphiB)
%             %    isoB =
%                minphiC = min(phi2C(:));
%                maxphiC = max(phi2C(:));
%                isoC = minphiC+0.7*(maxphiC-minphiC)
%             %    isoB =
% 
%                figure(2)
% 
%                imagesc(phi2A)
%                colormap;
%                set(gcf, 'color', 'white')
%                title('phi A');
%                colorbar %('location','southoutside')
%                axis off
%                axis equal
%                saveName = sprintf('phiA.[%.2f.%.2f.%.2f].[%.4f.%.4f.%.4f.%.4f]-[%d]',...
%                  KN(1), KN(2), KN(3), fnA, fnB, fnC, fnD, Nd);
%                title(saveName);
%                set(gcf, 'unit', 'normalized', 'position', [0.05, 0.1, 0.8, 0.8])
%                f = getframe(gcf);
%                imwrite(f.cdata,  ['figure/', saveName, '.png']);
% 
%                figure(3)
%                imagesc(phi2B)
%                colormap;
%                set(gcf, 'color', 'white')
%                title('phi B');
%                colorbar %('location','southoutside')
%                axis off
%                axis equal;
%                saveName = sprintf('phiB.[%.2f.%.2f.%.2f].[%.4f.%.4f.%.4f.%.4f]-[%d]',...
%                  KN(1), KN(2), KN(3), fnA, fnB, fnC, fnD, Nd);
%                title(saveName);
%                set(gcf, 'unit', 'normalized', 'position', [0.05, 0.1, 0.8, 0.8])
%                f = getframe(gcf);
%                imwrite(f.cdata,  ['figure/', saveName, '.png']);
%                
%                figure(4)
%                imagesc(phi2C)
%                colormap;
%                set(gcf, 'color', 'white')
%                title('phi C');
%                colorbar %('location','southoutside')
%                axis off
%                axis equal;
%                saveName = sprintf('phiC.[%.2f.%.2f.%.2f].[%.4f.%.4f.%.4f.%.4f]-[%d]',...
%                  KN(1), KN(2), KN(3), fnA, fnB, fnC, fnD, Nd);
%                title(saveName);
%                set(gcf, 'unit', 'normalized', 'position', [0.05, 0.1, 0.8, 0.8])
%                f = getframe(gcf);
%                imwrite(f.cdata,  ['figure/', saveName, '.png']);
%                
%                figure(5)
%                imagesc(phi2D)
%                colormap;
%                set(gcf, 'color', 'white')
%                title('phi D');
%                colorbar %('location','southoutside')
%                axis off
%                axis equal;
%                saveName = sprintf('phiD.[%.2f.%.2f.%.2f].[%.4f.%.4f.%.4f.%.4f]-[%d]',...
%                  KN(1), KN(2), KN(3), fnA, fnB, fnC, fnD, Nd);
%                title(saveName);
%                set(gcf, 'unit', 'normalized', 'position', [0.05, 0.1, 0.8, 0.8])
%                f = getframe(gcf);
%                imwrite(f.cdata,  ['figure/', saveName, '.png']);
%             end
         
%%%  谱空间分布
%             char parameters;
%             parameters = sprintf('[fA, fB, fC, fD]: [%f,%f,%f,%f]', fA, fB, fC,fD);
%             fprintf('parameters: %s\n', parameters);
% 
%             fnA = fA / 100;
%             fnB = fB / 100;
%             fnC = fC / 100;
%             fnD = fD / 100;
%             srcfilename = sprintf('vec.phiA.[%.2f.%.2f.%.2f].[%.4f.%.4f.%.4f.%.4f]-[%d].txt',...
%                 KN(1), KN(2), KN(3), fnA, fnB, fnC, fnD, Nd);
% 
%             vecA = load(strrep(srcfilename, 'A', 'A'));
%             vecB = load(strrep(srcfilename, 'A', 'B1'));
%             vecC = load(strrep(srcfilename, 'A', 'C'));
%             vecD = load(strrep(srcfilename, 'A', 'B2'));
%             
%             figure(6)
%             hold on
%             x = vecA(:,1);
%             y = vecA(:,2);
%             index = vecA(:,3);
%             n = length(index);
%             for i = 1:1:n
%                if index(i) > 1.0e-1
%                    plot(x(i),y(i),'MarkerFaceColor',[1 0 0], ...
%                    'MarkerEdgeColor',[1 0 0],'Marker','o', ...
%                    'MarkerSize', 10)
%                elseif (index(i) < 1.0e-1 && index(i) > 1.0e-2)
%                    plot(x(i),y(i),'MarkerFaceColor',[0 0 1], ...
%                    'MarkerEdgeColor',[0 0 1],'Marker','o', ...
%                    'MarkerSize', 8)
%                elseif (index(i) < 1.0e-2 && index(i) > 1.0e-3)
%                    plot(x(i),y(i),'MarkerFaceColor',[0 1 0], ...
%                    'MarkerEdgeColor',[0 1 0],'Marker','o', ...
%                    'MarkerSize', 6.5)
%             %    elseif (index(i) < 1.0e-4 && index(i) > 1.0e-5)
%             %        plot(x(i),y(i),'MarkerFaceColor',[112/255 128/255 105/255], ...
%             %        'MarkerEdgeColor',[112/255 128/255 105/255],'Marker','o', ...
%             %        'MarkerSize', 5.5)
%             %    elseif (index(i) < 1.0e-5 && index(i) > 1.0e-6)
%             %        plot(x(i),y(i),'MarkerFaceColor',[8/255 46/255 84/255], ...
%             %        'MarkerEdgeColor',[8/255 46/255 84/255],'Marker','o', ...
%             %        'MarkerSize', 4.5)
%             %    elseif (index(i) < 1.0e-6 && index(i) > 1.0e-10)
%             %        plot(x(i),y(i),'MarkerFaceColor',[0 0 0], ...
%             %        'MarkerEdgeColor',[0 0 0],'Marker','o', ...
%             %        'MarkerSize', 3)
%                end
%             end
%             axis off
%             axis equal;
%             colormap;
%             set(gcf, 'color', 'white')    
%             title('Plane waves of phiA')
%             xlabel('')
%             ylabel('')
%             saveName = sprintf('phiA_fft.[%.2f.%.2f.%.2f].[%.4f.%.4f.%.4f.%.4f]-[%d]',...
%                  KN(1), KN(2), KN(3), fnA, fnB, fnC, fnD, Nd);
%                title(saveName);
%                set(gcf, 'unit', 'normalized', 'position', [0.05, 0.1, 0.8, 0.8])
%                f = getframe(gcf);
%                imwrite(f.cdata,  ['figure/', saveName, '.png']);
%             
%                
%             figure(7)
%             hold on
%             x = vecB(:,1);
%             y = vecB(:,2);
%             index = vecB(:,3);
%             n = length(index);
%             for i = 1:1:n
%                if index(i) > 1.0e-1
%                    plot(x(i),y(i),'MarkerFaceColor',[1 0 0], ...
%                    'MarkerEdgeColor',[1 0 0],'Marker','o', ...
%                    'MarkerSize', 10)
%                elseif (index(i) < 1.0e-1 && index(i) > 1.0e-2)
%                    plot(x(i),y(i),'MarkerFaceColor',[0 0 1], ...
%                    'MarkerEdgeColor',[0 0 1],'Marker','o', ...
%                    'MarkerSize', 8)
%                elseif (index(i) < 1.0e-2 && index(i) > 1.0e-3)
%                    plot(x(i),y(i),'MarkerFaceColor',[0 1 0], ...
%                    'MarkerEdgeColor',[0 1 0],'Marker','o', ...
%                    'MarkerSize', 6.5)
%             %    elseif (index(i) < 1.0e-4 && index(i) > 1.0e-5)
%             %        plot(x(i),y(i),'MarkerFaceColor',[112/255 128/255 105/255], ...
%             %        'MarkerEdgeColor',[112/255 128/255 105/255],'Marker','o', ...
%             %        'MarkerSize', 5.5)
%             %    elseif (index(i) < 1.0e-5 && index(i) > 1.0e-6)
%             %        plot(x(i),y(i),'MarkerFaceColor',[8/255 46/255 84/255], ...
%             %        'MarkerEdgeColor',[8/255 46/255 84/255],'Marker','o', ...
%             %        'MarkerSize', 4.5)
%             %    elseif (index(i) < 1.0e-6 && index(i) > 1.0e-10)
%             %        plot(x(i),y(i),'MarkerFaceColor',[0 0 0], ...
%             %        'MarkerEdgeColor',[0 0 0],'Marker','o', ...
%             %        'MarkerSize', 3)
%                end
%             end
%             axis off
%             axis equal;
%             set(gcf, 'color', 'white')    
%             colormap;
%             title('Plane waves of phiB')
%             xlabel('')
%             ylabel('')
%             saveName = sprintf('phiB_fft.[%.2f.%.2f.%.2f].[%.4f.%.4f.%.4f.%.4f]-[%d]',...
%                  KN(1), KN(2), KN(3), fnA, fnB, fnC, fnD, Nd);
%                title(saveName);
%                set(gcf, 'unit', 'normalized', 'position', [0.05, 0.1, 0.8, 0.8])
%                f = getframe(gcf);
%                imwrite(f.cdata,  ['figure/', saveName, '.png']);
%             
%                
%                figure(8)
%             hold on
%             x = vecC(:,1);
%             y = vecC(:,2);
%             index = vecC(:,3);
%             n = length(index);
%             for i = 1:1:n
%                if index(i) > 1.0e-1
%                    plot(x(i),y(i),'MarkerFaceColor',[1 0 0], ...
%                    'MarkerEdgeColor',[1 0 0],'Marker','o', ...
%                    'MarkerSize', 10)
%                elseif (index(i) < 1.0e-1 && index(i) > 1.0e-2)
%                    plot(x(i),y(i),'MarkerFaceColor',[0 0 1], ...
%                    'MarkerEdgeColor',[0 0 1],'Marker','o', ...
%                    'MarkerSize', 8)
%                elseif (index(i) < 1.0e-2 && index(i) > 1.0e-3)
%                    plot(x(i),y(i),'MarkerFaceColor',[0 1 0], ...
%                    'MarkerEdgeColor',[0 1 0],'Marker','o', ...
%                    'MarkerSize', 6.5)
%             %    elseif (index(i) < 1.0e-4 && index(i) > 1.0e-5)
%             %        plot(x(i),y(i),'MarkerFaceColor',[112/255 128/255 105/255], ...
%             %        'MarkerEdgeColor',[112/255 128/255 105/255],'Marker','o', ...
%             %        'MarkerSize', 5.5)
%             %    elseif (index(i) < 1.0e-5 && index(i) > 1.0e-6)
%             %        plot(x(i),y(i),'MarkerFaceColor',[8/255 46/255 84/255], ...
%             %        'MarkerEdgeColor',[8/255 46/255 84/255],'Marker','o', ...
%             %        'MarkerSize', 4.5)
%             %    elseif (index(i) < 1.0e-6 && index(i) > 1.0e-10)
%             %        plot(x(i),y(i),'MarkerFaceColor',[0 0 0], ...
%             %        'MarkerEdgeColor',[0 0 0],'Marker','o', ...
%             %        'MarkerSize', 3)
%                end
%             end
%             axis off
%             axis equal;
%             set(gcf, 'color', 'white')    
%             colormap;
%             title('Plane waves of phiC')
%             xlabel('')
%             ylabel('')
%             saveName = sprintf('phiC_fft.[%.2f.%.2f.%.2f].[%.4f.%.4f.%.4f.%.4f]-[%d]',...
%                  KN(1), KN(2), KN(3), fnA, fnB, fnC, fnD, Nd);
%                title(saveName);
%                set(gcf, 'unit', 'normalized', 'position', [0.05, 0.1, 0.8, 0.8])
%                f = getframe(gcf);
%                imwrite(f.cdata,  ['figure/', saveName, '.png']);
%             
%                
%                 figure(9)
%             hold on
%             x = vecD(:,1);
%             y = vecD(:,2);
%             index = vecD(:,3);
%             n = length(index);
%             for i = 1:1:n
%                if index(i) > 1.0e-1
%                    plot(x(i),y(i),'MarkerFaceColor',[1 0 0], ...
%                    'MarkerEdgeColor',[1 0 0],'Marker','o', ...
%                    'MarkerSize', 10)
%                elseif (index(i) < 1.0e-1 && index(i) > 1.0e-2)
%                    plot(x(i),y(i),'MarkerFaceColor',[0 0 1], ...
%                    'MarkerEdgeColor',[0 0 1],'Marker','o', ...
%                    'MarkerSize', 8)
%                elseif (index(i) < 1.0e-2 && index(i) > 1.0e-3)
%                    plot(x(i),y(i),'MarkerFaceColor',[0 1 0], ...
%                    'MarkerEdgeColor',[0 1 0],'Marker','o', ...
%                    'MarkerSize', 6.5)
%             %    elseif (index(i) < 1.0e-4 && index(i) > 1.0e-5)
%             %        plot(x(i),y(i),'MarkerFaceColor',[112/255 128/255 105/255], ...
%             %        'MarkerEdgeColor',[112/255 128/255 105/255],'Marker','o', ...
%             %        'MarkerSize', 5.5)
%             %    elseif (index(i) < 1.0e-5 && index(i) > 1.0e-6)
%             %        plot(x(i),y(i),'MarkerFaceColor',[8/255 46/255 84/255], ...
%             %        'MarkerEdgeColor',[8/255 46/255 84/255],'Marker','o', ...
%             %        'MarkerSize', 4.5)
%             %    elseif (index(i) < 1.0e-6 && index(i) > 1.0e-10)
%             %        plot(x(i),y(i),'MarkerFaceColor',[0 0 0], ...
%             %        'MarkerEdgeColor',[0 0 0],'Marker','o', ...
%             %        'MarkerSize', 3)
%                end
%             end
%             axis off
%             axis equal;
%             set(gcf, 'color', 'white')    
%             colormap;
%             title('Plane waves of phiD')
%             xlabel('')
%             ylabel('')
%             saveName = sprintf('phiD_fft.[%.2f.%.2f.%.2f].[%.4f.%.4f.%.4f.%.4f]-[%d]',...
%                  KN(1), KN(2), KN(3), fnA, fnB, fnC, fnD, Nd);
%                title(saveName);
%                set(gcf, 'unit', 'normalized', 'position', [0.05, 0.1, 0.8, 0.8])
%                f = getframe(gcf);
%                imwrite(f.cdata,  ['figure/', saveName, '.png']);
        end
    end
 end

%energy = load('hamilton.dat');
%plot(energy(:,4));
%set(gcf, 'color', 'white')
%saveName = sprintf('HC.hamilton.[30.00.30.00.30.00].fA-[0.%d-0.%d].fB-[0.%d-0.%d]-[128]', fALft, fARgt, fBLft, fBRgt);
%title(saveName);
%axis on
%set(gcf, 'unit', 'normalized', 'position', [0.05, 0.1, 0.8, 0.8])
%f = getframe(gcf);
%imwrite(f.cdata,  [saveName, '.png']);
%pause(1)
