using UnityEngine;

namespace ChaosRL
{
    public class ArenaGridSpawner : MonoBehaviour
    {
        //------------------------------------------------------------------
        [Header( "Grid Settings" )]
        [SerializeField] private GameObject _arenaPrefab;
        [SerializeField] private float _spacing = 20f;

        [Header( "Spawn Settings" )]
        [SerializeField] private bool _centerGrid = true;
        [SerializeField] private Vector3 _offset = Vector3.zero;

        private Vector3Int _gridSize;
        private int _numberOfArenas;
        //------------------------------------------------------------------
        private void Start()
        {
            CalculateGridSize();
            SpawnArenaGrid();
        }
        //------------------------------------------------------------------
        private void CalculateGridSize()
        {
            _numberOfArenas = Academy.Instance.NumEnvs;
            // Calculate grid dimensions to fit the number of arenas in a 3D cube
            // Creates as close to a cube shape as possible
            int sideLength = Mathf.CeilToInt( Mathf.Pow( _numberOfArenas, 1f / 3f ) );
            _gridSize = new Vector3Int( sideLength, sideLength, sideLength );
        }
        //------------------------------------------------------------------
        private void SpawnArenaGrid()
        {
            if (_arenaPrefab == null)
            {
                Debug.LogError( "Arena prefab is not assigned!" );
                return;
            }

            Vector3 startPosition = transform.position + _offset;

            // Calculate center offset if centering is enabled
            if (_centerGrid)
            {
                Vector3 gridCenter = new Vector3(
                    (_gridSize.x - 1) * _spacing * 0.5f,
                    (_gridSize.y - 1) * _spacing * 0.5f,
                    (_gridSize.z - 1) * _spacing * 0.5f
                );
                startPosition -= gridCenter;
            }

            // Spawn arenas in a 3D grid
            int arenasSpawned = 0;
            for (int x = 0; x < _gridSize.x; x++)
            {
                for (int y = 0; y < _gridSize.y; y++)
                {
                    for (int z = 0; z < _gridSize.z; z++)
                    {
                        if (arenasSpawned >= _numberOfArenas)
                            break;

                        Vector3 spawnPosition = startPosition + new Vector3(
                            x * _spacing,
                            y * _spacing,
                            z * _spacing
                        );

                        GameObject arena = Instantiate( _arenaPrefab, spawnPosition, Quaternion.identity );
                        arena.name = $"Arena_{x}_{y}_{z}";
                        arenasSpawned++;
                    }
                    if (arenasSpawned >= _numberOfArenas)
                        break;
                }
                if (arenasSpawned >= _numberOfArenas)
                    break;
            }

            Debug.Log( $"Spawned {arenasSpawned} arenas in a {_gridSize.x}x{_gridSize.y}x{_gridSize.z} grid" );
        }
        //------------------------------------------------------------------
#if UNITY_EDITOR
        private void OnDrawGizmosSelected()
        {
            if (_arenaPrefab == null) return;

            Gizmos.color = Color.cyan;

            // Calculate temporary grid size for visualization
            int sideLength = Mathf.CeilToInt( Mathf.Pow( _numberOfArenas, 1f / 3f ) );
            Vector3Int tempGridSize = new Vector3Int( sideLength, sideLength, sideLength );

            Vector3 startPosition = transform.position + _offset;

            if (_centerGrid)
            {
                Vector3 gridCenter = new Vector3(
                    (tempGridSize.x - 1) * _spacing * 0.5f,
                    (tempGridSize.y - 1) * _spacing * 0.5f,
                    (tempGridSize.z - 1) * _spacing * 0.5f
                );
                startPosition -= gridCenter;
            }

            // Draw gizmos to visualize spawn points
            int arenasDrawn = 0;
            for (int x = 0; x < tempGridSize.x; x++)
            {
                for (int y = 0; y < tempGridSize.y; y++)
                {
                    for (int z = 0; z < tempGridSize.z; z++)
                    {
                        if (arenasDrawn >= _numberOfArenas)
                            break;

                        Vector3 spawnPosition = startPosition + new Vector3(
                            x * _spacing,
                            y * _spacing,
                            z * _spacing
                        );

                        Gizmos.DrawWireCube( spawnPosition, Vector3.one );
                        arenasDrawn++;
                    }
                    if (arenasDrawn >= _numberOfArenas)
                        break;
                }
                if (arenasDrawn >= _numberOfArenas)
                    break;
            }
        }
#endif
        //------------------------------------------------------------------
    }
}