from __future__ import annotations
from toolz import pipe, curried, curry

import functools
import json
from abc import ABC, ABCMeta
from typing import Iterator, Optional, Type
from weakref import WeakKeyDictionary

import pandas as pd
import pyproj
import requests
import shapely.geometry
import shapely.geometry
import shapely.ops
from geopandas import GeoSeries
from toolz import curry, pipe
from tile2net.raster import util

from tile2net.logger import logger


if False:
    from tile2net.raster.tile import Tile

class SourceMeta(ABCMeta):
    catalog: dict[str, Type[Source]] = {}

    @classmethod
    @property
    def coverage(cls) -> GeoSeries:
        coverages: list[GeoSeries] = []
        for source in cls.catalog.values():
            try:
                axis = pd.Index([source.name] * len(source.coverage), name='source')
                coverage = (
                    source.coverage
                    .set_crs('epsg:4326')
                    .set_axis(axis)
                )
            except Exception as e:
                logger.error(
                    f'Could not get coverage for {source.name}, skipping:\n'
                    f'{e}'
                )
            else:
                coverages.append(coverage)

        coverage = pd.concat(coverages)
        cls.coverage = coverage
        return coverage

    def __getitem__(
        cls: Type[Source],
        item: list[float] | str | shapely.geometry.base.BaseGeometry,
    ) -> Optional['Source']:
        # todo: index index for which sources contain keyword
        original = item
        matches: GeoSeries = (
            cls.__class__.coverage.geometry
            .to_crs(3857)
        )
        if isinstance(item, list):
            s, w, n, e = item
            display_name = util.reverse_geocode(item).casefold()
            # noinspection PyTypeChecker
            coverage: GeoSeries = SourceMeta.coverage
            index = set(coverage.index)
            loc = [
                source.name
                for source in cls.catalog.values() if
                source.name in index
                and source.keyword.casefold() in display_name
            ]
            matches = matches.loc[loc]
            if matches.empty:
                logger.warning(
                    f'No source was found to have a matching keyword with {display_name}'
                )
            item = shapely.geometry.box(w, s, e, n)

        if isinstance(item, shapely.geometry.base.BaseGeometry):
            trans = pyproj.Transformer.from_crs(
                'epsg:4326', 'epsg:3857', always_xy=True
            ).transform
            item = shapely.ops.transform(trans, item)

            loc = matches.intersects(item)
            matches = matches.loc[loc]
            if matches.empty:
                return None
                # raise KeyError(f'No source found for {item}')
            items = (
                matches.intersection(item)
                .area
                .__truediv__(matches.area)
                # .idxmax()
            )
            if len(items) > 1:
                logger.info(
                    f'Found multiple sources for the location, in descending IOU: '
                    f'{items.sort_values(ascending=False).index.tolist()}'
                )
            item = items.idxmax()

        if isinstance(item, str):
            if item not in cls.catalog:
                return None
                # raise KeyError(f'No source found for {item}')
            source = cls.catalog[item]

        else:
            raise TypeError(f'Invalid type {type(original)} for {original}')
        return source()

    def __init__(self: Type[Source], name, bases, attrs, **kwargs):
        # super(type(self), self).__init__(name, bases, attrs, **kwargs)
        super().__init__(name, bases, attrs)
        if (
                ABC not in bases
                and kwargs.get('init', True)
        ):
            if self.name is None:
                raise ValueError(f'{self} must have a name')
            if self.name in self.catalog:
                raise ValueError(f'{self} name already in use')
            self.catalog[self.name] = self

class Source(ABC, metaclass=SourceMeta):
    name: str = None  # name of the source
    coverage: GeoSeries = None  # coverage that contains a polygon representing the coverage
    zoom: int = None  # xyz tile zoom level
    extension = 'png'
    tiles: str = None
    tilesize: int = 256  # pixels per tile side
    keyword: str  # required match when reverse geolocating address from point

    def __getitem__(self, item: Iterator[Tile]):
        tiles = self.tiles
        yield from (
            tiles.format(z=tile.zoom, y=tile.ytile, x=tile.xtile)
            for tile in item
        )

    def __bool__(self):
        return True

    def __repr__(self):
        return f'<{self.__class__.__qualname__} {self.name} at {hex(id(self))}>'

    def __str__(self):
        return self.name

    def __init_subclass__(cls, **kwargs):
        # complains if gets kwargs
        super().__init_subclass__()

    def __eq__(self, other):
        if isinstance(other, str):
            return self.name == other
        if isinstance(other, Source):
            return self.__class__ == other.__class__
        return NotImplemented

class class_attr:
    # caches properties to the class if class is not abc
    cache = WeakKeyDictionary()

    @classmethod
    def relevant_to(cls, item: SourceMeta) -> set[class_attr]:
        res = {
            attr
            for subclass in item.mro()
            if subclass in cls.cache
            for attr in cls.cache[subclass]
        }
        return res

    def __get__(self, instance, owner: Source | SourceMeta):
        result = self.func(owner)
        type.__setattr__(owner, self.name, result)
        return result

    def __init__(self, func):
        if not isinstance(func, property):
            raise TypeError(f'{func} must be a property')
        func = func.fget
        self.func = func
        functools.update_wrapper(self, func)
        return

    def __set_name__(self, owner, name):
        self.name = name
        self.cache.setdefault(owner, set()).add(self)

    def __repr__(self):
        return f'<{self.__class__.__name__} {self.__name__}>'

    def __hash__(self):
        return hash(self.name)

    def __eq__(self, other):
        return self.name == other.name

# noinspection PyPropertyDefinition
class ArcGis(Source, ABC):
    server: str = None

    @class_attr
    @property
    def layer_info(cls):
        response = requests.get(cls.metadata)
        response.raise_for_status()
        text = response.text
        try:
            res = json.loads(text)
        except json.JSONDecodeError as e:
            logger.error(f'Could not parse JSON from stdin: {e}')
            logger.error(f'{cls.metadata=}; {cls.server=}')
            logger.error(f'JSON: {text}')
            raise
        return res

    @class_attr
    @property
    def coverage(cls):
        crs = cls.layer_info['spatialReference']['latestWkid']
        res = GeoSeries([shapely.geometry.box(
            cls.layer_info['fullExtent']['xmin'],
            cls.layer_info['fullExtent']['ymin'],
            cls.layer_info['fullExtent']['xmax'],
            cls.layer_info['fullExtent']['ymax'],
        )], crs=crs).to_crs('epsg:4326')
        return res

    @class_attr
    @property
    def zoom(cls):
        try:
            res = cls.layer_info['maxLOD']
        except KeyError:
            res = max(cls.layer_info['tileInfo']['lods'], key=lambda x: x['level'])['level']
        res = min(res, 20)
        return res

    @class_attr
    @property
    def metadata(cls):
        return cls.server + '?f=json'

    @class_attr
    @property
    def tiles(cls):
        return cls.server + '/tile/{z}/{y}/{x}'

class NewYorkCity(ArcGis):
    server = 'https://tiles.arcgis.com/tiles/yG5s3afENB5iO9fj/arcgis/rest/services/NYC_Orthos_-_2020/MapServer'
    name = 'nyc'
    keyword = 'New York'

class NewYork(ArcGis):
    server = 'https://orthos.its.ny.gov/arcgis/rest/services/wms/2020/MapServer'
    name = 'ny'
    keyword = 'New York City'

class Massachusetts(ArcGis):
    server = 'https://tiles.arcgis.com/tiles/hGdibHYSPO59RG1h/arcgis/rest/services/USGS_Orthos_2019/MapServer'
    name = 'ma'
    keyword = 'Massachusetts'

class KingCountyWashington(ArcGis):
    server = 'https://gismaps.kingcounty.gov/arcgis/rest/services/BaseMaps/KingCo_Aerial_2021/MapServer'
    name = 'king'
    keyword = 'King'

class WashingtonDC(ArcGis):
    server = 'https://imagery.dcgis.dc.gov/dcgis/rest/services/Ortho/Ortho_2021/ImageServer'
    name = 'dc'
    tilesize = 512
    extension = 'jpeg'
    keyword = 'Columbia'

    def __getitem__(self, item: Iterator[Tile]):
        for tile in item:
            top, left, bottom, right = tile.transformProject(tile.crs, 3857)
            yield (
                f'https://imagery.dcgis.dc.gov/dcgis/rest/services/Ortho/Ortho_2021/ImageServer'
                f'/exportImage?f=image&bbox={bottom}%2C{right}%2C{top}%2C{left}'
                f'&imageSR=102100&bboxSR=102100&size=512%2C512'
            )

    @class_attr
    @property
    def zoom(cls):
        return 19
        # return 20

class LosAngeles(ArcGis):
    server = 'https://cache.gis.lacounty.gov/cache/rest/services/LACounty_Cache/LACounty_Aerial_2014/MapServer'
    name = 'la'
    keyword = 'Los Angeles'

    # to test case where a source raises an error due to metadata failure
    #   other sources should still function
    # @class_attr
    # @property
    # def metadata(cls):
    #     raise NotImplementedError

# class WestOregon(ArcGis, init=False):
# class WestOregon(ArcGis):
#     server = 'https://imagery.oregonexplorer.info/arcgis/rest/services/OSIP_2018/OSIP_2018_WM/ImageServer'
#     name = 'w_or'
#     extension = 'jpeg'
#     keyword = 'Oregon'
#     # todo: ssl incorrectly configured; come back later
#
# # class EastOregon(ArcGis, init=False):
# class EastOregon(ArcGis, init=False):
#
#     server = 'https://imagery.oregonexplorer.info/arcgis/rest/services/OSIP_2017/OSIP_2017_WM/ImageServer'
#     name = 'e_or'
#     extension = 'jpeg'
#     keyword = 'Oregon'

class Oregon(ArcGis):
    server = 'https://imagery.oregonexplorer.info/arcgis/rest/services/OSIP_2022/OSIP_2022_WM/ImageServer'
    name = 'or'
    extension = 'jpeg'
    keyword = 'Oregon'

class NewJersey(ArcGis):
    server = 'https://maps.nj.gov/arcgis/rest/services/Basemap/Orthos_Natural_2020_NJ_WM/MapServer'
    name = 'nj'
    keyword = 'New Jersey'

class SpringHillTN(ArcGis):
    server = 'https://tiles.arcgis.com/tiles/tF0XsRR9ptiKNVW2/arcgis/rest/services/Spring_Hill_Imagery_WGS84/MapServer'
    name = 'sh_tn'
    keyword = 'Spring Hill'


if __name__ == '__main__':
    from tile2net import Raster
    # when testing, comment out super().
    assert Raster(location='New Brunswick, New Jersey').source == 'nj'
    assert Raster(location='New York City').source == 'nyc'
    assert Raster(location='New York').source in ('nyc', 'ny')
    assert Raster(location='Massachusetts').source == 'ma'
    assert Raster(location='King County, Washington').source == 'king'
    assert Raster(location='Washington, DC', zoom=19).source == 'dc'
    assert Raster(location='Los Angeles', zoom=19).source == 'la'
    assert Raster(location='Jersey City', zoom=19).source == 'nj'
    assert Raster(location='Hoboken', zoom=19).source == 'nj'
    assert Raster(location="Spring Hill, TN", zoom=20).source == "sh_tn"

